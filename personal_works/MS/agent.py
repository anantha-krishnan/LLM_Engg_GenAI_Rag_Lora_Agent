# agent.py
import os
import subprocess
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain.chains import create_history_aware_retriever
from langchain.docstore.document import Document
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_core.pydantic_v1 import BaseModel, Field

import global_vars

# --- AGENT SETUP ---
DB_DIR = (Path(__file__).parent / "chroma_db_mdl").as_posix()
LLM_MODEL = global_vars.model_openai_4omini
SELECTION_KEYWORDS = ["select", "choose", "pick", "option", "go with", "take", "i want", "i'll have"]
COMPONENTS_SELECTED = [{"full_file_path": "", "system_def_name": ""}]
EDIT_KEYWORDS = ["adjust", "modify", "change", "edit", "optimize", "tune"]
EXIT_EDIT_KEYWORDS = ["done editing", "exit edit mode", "finish editing", "stop editing", "back to finding", "go back to finding"]

class ComponentSelection(BaseModel):
    """
    Model to represent a selected component with its full file path and system_def_name.
    """
    full_file_path: str = Field(description="The full file path of the selected component. It's value should be obtained from the 'full_file_path' field in the chat history.")
    system_def_name: str = Field(description="The system_def_name of the selected component. It's value should be obtained from the 'system_def_name' field in the chat history.")

class MdlAgent:
    def __init__(self):
        # enum all available conversation modes
        self.conversation_modes = {"FINDING": "FindingMode", "EDITING": "EditingMode"}
        self.conversation_mode = self.conversation_modes["FINDING"] # default mode
        self.active_component_context={"content":""}
        if not os.path.exists(DB_DIR):
            raise FileNotFoundError(
                f"Vector store not found at {DB_DIR}. "
                "Please run `python ingest.py` first."
            )
        knn_k=500
        self.vector_store = Chroma(
            persist_directory=DB_DIR, 
            embedding_function=OpenAIEmbeddings()
        )
        semantic_retriever = self.vector_store.as_retriever(search_kwargs={'k': knn_k})
        print("Initializing BM25 Retriever...")
        all_docs_from_db = self.vector_store.get(include=["metadatas", "documents"])
        
        # Reconstruct the Document objects for the BM25 retriever
        reconstructed_docs = []
        for i, doc_text in enumerate(all_docs_from_db['documents']):
            reconstructed_docs.append(
                Document(
                    page_content=doc_text, 
                    metadata=all_docs_from_db['metadatas'][i]
                )
            )
        
        bm25_retriever = BM25Retriever.from_documents(reconstructed_docs)
        bm25_retriever.k = knn_k # Match the k value for consistency

        # 3. Initialize the EnsembleRetriever
        # This retriever combines the results of the other two
        # The weights determine how much influence each retriever has. 0.5/0.5 is a good start. tune it as per requirement.
        # You can also adjust the k value for each retriever individually if needed.
        # The EnsembleRetriever will return up to k results total, combining results from both retrievers.
        self.retriever = EnsembleRetriever(
            retrievers=[semantic_retriever, bm25_retriever], weights=[0.5, 0.5]
        )
        print("Hybrid search (EnsembleRetriever) is ready.")

        # --- END OF NEW SETUP ---
        self.llm = ChatOpenAI(model=LLM_MODEL, temperature=0.3, streaming=True)
        # We need a non-streaming and non creative LLM for the JSON extraction chain
        self.extraction_llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
        self.extraction_chain = self._create_extraction_chain()
        self.chain = self._create_rag_chain()
        self.editor_chain = self._create_editor_chain()
    def open_motion_view(self):
        """
        Function to load the selected component into MotionView.
        This function is called after all the selections have been made. It opens MotionView and loads the selected components.
        Args:
            None
        Returns:
            None
        """
        with open("open_mv_model.py", "w") as f:
            f.write("from hw import mview\n")
            f.write("from pathlib import Path\n\n")
            for comp in COMPONENTS_SELECTED:
                if comp["full_file_path"] and comp["system_def_name"]:
                    f.write(f'mview.System (definition_file = "{str(comp["full_file_path"])}", name="{Path(comp["full_file_path"]).stem}",  definition_name = "{comp["system_def_name"]}")\n')
                    
        # start a cmd command using subprocess to open MotionView with the selected components in the command line
        subprocess.Popen([
            "C:/Altair_Installs/2026_0_10_release/hwdesktop/hwx/bin/win64/runhwx.exe",
            "-client", "HyperWorksDesktop",
            "-plugin", "HyperworksPost",
            "-profile", "HyperworksPost",
            "-clientconfig", "hwmbdmodel.dat",
            "-python", "D:/PaperWork/personal/AI/LLM_Engg_GenAI_Rag_Lora_Agent/personal_works/MS/open_mv_model.py"
        ])
    def _add_selected_component(self, extracted_data: dict):
        if extracted_data and extracted_data.get("full_file_path"):
            file_path = extracted_data.get("full_file_path", "N/A")
            sys_def = extracted_data.get("system_def_name", "N/A")
            COMPONENTS_SELECTED.append({
                "full_file_path": Path(file_path).as_posix(),
                "system_def_name": sys_def
            })
            print(f"[INFO] Extracted selection - full_file_path: {file_path}, SystemDefName: {sys_def}")
            return f"Confirmed: Added '{file_path}' to your selections. What would you like to find next?"
        else:
            print("[INFO] Extraction chain did not find a valid selection.")
            return None
    def _read_mdl_file_content(self, file_path: str) -> str:
        """Helper function to read the content of an MDL file."""
        try:
            with open(file_path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            print(f"[ERROR] File not found: {file_path}")
            return "Error: Could not read the file."
        except Exception as e:
            print(f"[ERROR] Error reading file {file_path}: {e}")
            return f"Error: An unexpected error occurred while reading the file: {e}"
    def _handle_potential_selection(self, user_input: str, chat_history: list):
        """
        Uses the extraction chain to identify if the user is making a selection.
        """
        # Quick check for keywords to avoid unnecessary LLM calls
        if not any(keyword in user_input.lower() for keyword in SELECTION_KEYWORDS):
            return None # Not a selection, proceed to RAG

        print("\n[INFO] Potential selection detected. Running extraction chain...")
        
        try:
            # Invoke the extraction chain
            extracted_data = self.extraction_chain.invoke({
                "chat_history": chat_history,
                "user_input": user_input
            })
            if extracted_data and extracted_data.get("full_file_path"):
                return extracted_data
            else:
                print("[INFO] Extraction chain did not find a valid selection.")
                return None # No valid selection found, proceed to RAG

        except Exception as e:
            print(f"[ERROR] Could not process selection: {e}")
            # Fall through to the RAG chain
            
        return None # If extraction fails or finds nothing, return None
    def _format_docs(self, docs):
        """Formats the retrieved documents for the prompt context."""
        if not docs:
            return "No relevant components were found in the library."
        
        formatted_docs = []
        for i, doc in enumerate(docs):
            meta = doc.metadata
            description_text = "N/A"
            if 'Description: ' in doc.page_content:
                 description_text = doc.page_content.split('Description: ')[-1]

            doc_str = (
                f"--- Component {i+1} ---\n"
                f"full_file_path: {meta.get('file_path', 'N/A')}\n"
                f"system_def_name: {meta.get('system_def_name', 'N/A')}\n"
                f"Type: {meta.get('model_type', 'N/A')}\n"
                f"Side: {meta.get('side', 'N/A')}\n"
                f"Category: {meta.get('category1', '')} {meta.get('category2', '')}\n"
                f"Description: {description_text}"
            )
            formatted_docs.append(doc_str)
        return "\n\n".join(formatted_docs)
    
    def _create_editor_chain(self):
        """Creates a chain specifically for understanding and modifying MDL file content."""
        
        editor_system_prompt = """
        You are an expert CAE assistant for Altair MotionView, specializing in editing component parameters defined in MDL files.
        Your task is to help a user understand and modify a specific entity within the provided MDL file content.

        Here is your workflow:
        1.  **Analyze the Request:** Based on the user's request, identify the specific entity they want to modify (e.g., "shock damper", "front bushing", "rod mass").
        2.  **Locate the Definition:** Scan the provided "MDL File Content" to find the corresponding definition block. For example, a "shock damper" is likely defined by a `*SetCoilSpring` statement, and a body's mass by `*SetBody`.
        3.  **Extract and Explain:**
            - Extract the current parameter values from the relevant line. For `*SetCoilSpring(dmp, LEFT, K, C, Fo, Lo)`, you would extract K (stiffness) and C (damping).
            - In simple terms, explain what these parameters mean in a physical context. For example: "The current stiffness (K) is ... A higher value makes the suspension stiffer. The current damping (C) is ... A higher value makes the shock settle faster after a bump."
        4.  **Guide the User:** Ask the user what changes they'd like to make. Offer suggestions like, "Would you like to make it softer or stiffer? Faster or slower damping?"
        5.  **Generate the New Statement:** Once the user provides the new values, generate the **complete and exact** `*Set...` statement with the updated values, preserving the original format. For example: `*SetCoilSpring(dmp,     LEFT,           10,          1,           0,         0)`.
        
        **Important Rules:**
        - ALWAYS base your analysis on the provided "MDL File Content".
        - If the user asks to modify something you can't find, inform them clearly.
        - Be conversational and helpful.
        **Formatting Rule**:
        - When you include mathematical equations or formulas, you MUST enclose them in standard LaTeX delimiters. Use `$$...$$` for equations on their own line (display mode) and `$...$` for equations within a line of text (inline mode).
        """
        
        editor_prompt = ChatPromptTemplate.from_messages([
            ("system", editor_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            # providing the file content directly in the context
            ("user", "My request is: '{user_request}'\n\n--- MDL File Content ---\n{mdl_content}")
        ])
        
        # use a standard LLM
        return editor_prompt | self.llm | StrOutputParser()

    def _create_rag_chain(self):
        """
        Creates the main RAG chain using LangChain Expression Language (LCEL).
        """
        contextualize_q_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )
        
        history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, contextualize_q_prompt
        )
        system_prompt = """
        You are an expert assistant for Altair MotionView with two primary roles: a Component Finder and a Concept Explainer.

        1.  **As a Component Finder**: When the user asks to find, list, or show components (e.g., "what suspensions do you have?", "find a macpherson strut"), your primary goal is to use the provided CONTEXT.
            - You MUST answer based ONLY on the component information in the CONTEXT section.
            - If the user asks a broad question (e.g., "what suspensions do you have?"), summarize the available components based on the context. Group them logically (e.g., by Component types and/or by side so on).
            - If the user asks a specific question (e.g., "find a macpherson strut"), list the specific components that match from the context.
            - If the context is empty, state that you could not find any matching components and suggest they broaden their search.
            - If the user asks a follow-up question, use the conversation history and the new context to provide a relevant answer.
            - You must have the full_file_path and system_def_name for each component listed.
            - Try to provide a summary of the components and a helpful idea to choose one of them and proceed.

        2.  **As a Concept Explainer**: If the user asks for an explanation of a technical term or concept (e.g., "what is a macpherson strut?", "explain rack and pinion steering"), you are free to use your general knowledge.
            - Provide a clear, helpful, and concise explanation of the concept.
            - You can do this before or after listing components from the CONTEXT if the user's query is mixed. For example, if they ask "show me macpherson struts and explain what they are".
            - You can also use the context to see if any components match the concept being explained, but your explanation should not rely solely on the context.
        
        **Formatting Rule**:
            - When you include mathematical equations or formulas, you MUST enclose them in standard LaTeX delimiters. Use `$$...$$` for equations on their own line (display mode) and `$...$` for equations within a line of text (inline mode).

        Always be helpful and differentiate between information from the library (CONTEXT) and your general knowledge.
        """
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "input: {input}\n\n--- CONTEXT ---\n{context}")
        ])
        
        def format_docs_in_chain(chain_input):
            docs = chain_input['context']
            formatted_context = self._format_docs(docs)
            return {
                "input": chain_input['input'],
                "chat_history": chain_input['chat_history'],
                "context": formatted_context
            }

        final_chain = (
            RunnablePassthrough.assign(context=history_aware_retriever)
            | format_docs_in_chain
            | qa_prompt
            | self.llm
            | StrOutputParser()
        )

        return final_chain
    def _create_extraction_chain(self):
        """
        Creates a dedicated chain to extract selection details from chat history.
        """
        parser = JsonOutputParser(pydantic_object=ComponentSelection)

        system_prompt = """
        You are an expert data extractor. Your task is to analyze a conversation history and a user's final selection statement.
        The history contains numbered lists of components with details like 'full_file_path'and 'system_def_name'.
        
        Based on the user's final selection, identify the component they have chosen.
        Extract ONLY the full file path from the 'full_file_path' field and the system_def_name from the 'system_def_name' field of the selected component.

        You MUST respond in a JSON format with two keys: "full_file_path" and "system_def_name".
        {format_instructions}
        - If the user selects by number (e.g., "option 2"), find component #2 in the last list provided. 
        - If the user selects by name (e.g., "the front sla suspension"), find the component that matches that description.
        - If you cannot determine a clear and unambiguous selection, respond with an empty JSON object {{}}.
        """
        
        extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            # The history will be passed in here
            MessagesPlaceholder(variable_name="chat_history"),
            # The user's final input that we are analyzing
            ("human", "User's final selection statement: '{user_input}'. Now, provide the JSON."),
        ])
        final_extraction_prompt = extraction_prompt.partial(format_instructions=parser.get_format_instructions())
        return final_extraction_prompt | self.extraction_llm | JsonOutputParser()
    def _set_up_editing_mode(self, user_input: str, chat_history: list):
        component_to_edit = self._handle_potential_selection(user_input, chat_history)
        if not component_to_edit:
            if not COMPONENTS_SELECTED:
                message = "No components have been selected yet. Please select a component first before attempting to edit."
                return self.yield_simple_string(message)
            else:
                message = "The selected components available for editing are:\n"
                for i, component in enumerate(COMPONENTS_SELECTED):
                    if "full_file_path" in component and "system_def_name" in component:
                        message += f"{i + 1}. {component['full_file_path']}\n"
                message += "Please select one of the above components to edit."
                return self.yield_simple_string(message)
        print(f"[INFO] Preparing to edit component: {component_to_edit.get('full_file_path')}")
        # Read the MDL file content
        f = self._read_mdl_file_content(component_to_edit.get("full_file_path"))
        if "Error:" in f:
            return self.yield_simple_string(f)
        self.active_component_context['content'] = f
        message = f"Loaded component '{component_to_edit.get('full_file_path')}' for editing.\n"                    
        self.conversation_mode = self.conversation_modes["EDITING"]
        return self.editor_chain.stream({
            "user_request": user_input,
            "mdl_content": self.active_component_context['content'],
            "chat_history": chat_history
        })
        
            
    def process_message(self, user_input: str, chat_history: list):
        """
        Processes the user's input.
        First, it checks for the command to open MotionView.
        Second, it checks for a selection command.
        Third, it checks for an edit command.
        If none is found, it invokes the RAG chain.
        """
        if 'open model in mv' in user_input.lower():
            self.open_motion_view()
            return "Opening MotionView with selected components..."
        
        # 2. Check for a selection command
        if 'i choose' in user_input.lower():
            selection_response = self._handle_potential_selection(user_input, chat_history)
            selection_response = self._add_selected_component(selection_response)
            if selection_response:
                # If it was a selection, return the confirmation message directly
                # We wrap it in a generator to be consistent with the streaming output
                def message_generator():
                    yield selection_response
                return message_generator()
        # check for exit edit mode command from user
        if any(keyword in user_input.lower() for keyword in EXIT_EDIT_KEYWORDS):
            if self.conversation_mode == self.conversation_modes["EDITING"]:
                self.conversation_mode = self.conversation_modes["FINDING"]
                self.active_component_context={"content":""}
                print("[STATE] Exiting EDITING mode. Returning to FINDING mode.")
                return self.yield_simple_string("Exited editing mode. You can now continue finding components.")
            else:
                return self.yield_simple_string("You are not in editing mode. No action taken.")
        # 3. Check for an edit command
        if self.conversation_mode == self.conversation_modes["EDITING"]:
            print("[STATE] In EDITING mode. Sending to editor chain.")
            # The user is continuing the editing conversation
            return self.editor_chain.stream({
                "user_request": user_input,
                "mdl_content": self.active_component_context['content'],
                "chat_history": chat_history
            })
        if any(keyword in user_input.lower() for keyword in EDIT_KEYWORDS):            
            return self._set_up_editing_mode(user_input, chat_history)
        # default to the RAG chain
        return self.invoke(user_input, chat_history)
    
    def invoke(self, input: str, chat_history: list):
        """Invokes the RAG chain with the user's question and history."""
        return self.chain.stream({
            "input": input,
            "chat_history": chat_history
        })
    def yield_simple_string(self, text: str):
        """
        Utility to yield a simple string as a generator.
        Args:
            text (str): The text to yield.
        Returns:
            Generator that yields the text.
        """
        yield text
# Create a single instance of the agent to be used by the UI
mdl_agent = MdlAgent()