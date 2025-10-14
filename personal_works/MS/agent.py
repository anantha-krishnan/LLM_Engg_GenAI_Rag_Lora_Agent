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
DB_DIR = "./chroma_db_mdl"
LLM_MODEL = global_vars.model_openai_4omini
SELECTION_KEYWORDS = ["select", "choose", "pick", "option", "go with", "take", "i want", "i'll have"]
COMPONENTS_SELECTED = [{"full_file_path": "", "system_def_name": ""}]

class ComponentSelection(BaseModel):
    """
    Model to represent a selected component with its full file path and system_def_name.
    """
    full_file_path: str = Field(description="The full file path of the selected component. It's value should be obtained from the 'full_file_path' field in the chat history.")
    system_def_name: str = Field(description="The system_def_name of the selected component. It's value should be obtained from the 'system_def_name' field in the chat history.")

class MdlAgent:
    def __init__(self):
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
                    f.write(f'mview.System (definition_file = "{str(comp["full_file_path"])}",  definition_name = "{comp["system_def_name"]}")\n')
                    
        # start a cmd command using subprocess to open MotionView with the selected components in the command line
        subprocess.Popen([
            "C:/Altair_Installs/2026_0_10_release/hwdesktop/hwx/bin/win64/runhwx.exe",
            "-client", "HyperWorksDesktop",
            "-plugin", "HyperworksPost",
            "-profile", "HyperworksPost",
            "-clientconfig", "hwmbdmodel.dat",
            "-python", "D:/PaperWork/personal/AI/LLM_Engg_GenAI_Rag_Lora_Agent/personal_works/MS/open_mv_model.py"
        ])
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
                file_path = extracted_data.get("full_file_path", "N/A")
                sys_def = extracted_data.get("system_def_name", "N/A")
                COMPONENTS_SELECTED.append({
                    "full_file_path": Path(file_path).as_posix(),
                    "system_def_name": sys_def
                })
                print(f"[INFO] Extracted selection - full_file_path: {file_path}, SystemDefName: {sys_def}")
                return f"Confirmed: Added '{file_path}' to your selections. What would you like to find next?"
                # Find the full Document object from our last retrieval
                # This is crucial for accessing all metadata for the callback
                selected_doc = None
                for doc in self.last_retrieved_docs:
                    if doc.metadata.get("file_path") == file_path:
                        selected_doc = doc
                        break
                
                if selected_doc:
                    self.on_component_selected(selected_doc)
                    return f"Confirmed: Added '{file_path}' to your selections. What would you like to find next?"
                else:
                    # This can happen if the user tries to select from an old message
                    return "It seems you're trying to select a component from a previous search. Please ask for the list again before making a selection."
                
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
        You are an expert assistant for Altair MotionView. Your goal is to help users find vehicle dynamic components from a library.
        You must answer questions based ONLY on the context provided from the component library.
        
        How to behave:
        - If the user asks a broad question (e.g., "what suspensions do you have?"), summarize the available components based on the context. Group them logically (e.g., by Front/Rear).
        - If the user asks a specific question (e.g., "find a macpherson strut"), list the specific components that match from the context.
        - If the context is empty, state that you could not find any matching components and suggest they broaden their search.
        - If the user asks a follow-up question, use the conversation history and the new context to provide a relevant answer.
        - Be helpful, concise, and always use the information from the 'CONTEXT' section.
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
    
    def process_message(self, user_input: str, chat_history: list):
        """
        Processes the user's input.
        First, it checks for a selection command.
        If none is found, it invokes the RAG chain.
        """
        if 'open model in mv' in user_input.lower():
            self.open_motion_view()
            return "Opening MotionView with selected components..."
        # 2. Check for a selection command
        if 'i choose' in user_input.lower():
            selection_response = self._handle_potential_selection(user_input, chat_history)
            if selection_response:
                # If it was a selection, return the confirmation message directly
                # We wrap it in a generator to be consistent with the streaming output
                def message_generator():
                    yield selection_response
                return message_generator()

        # 2. If not a selection command, run the RAG chain
        return self.invoke(user_input, chat_history)
    
    def invoke(self, input: str, chat_history: list):
        """Invokes the RAG chain with the user's question and history."""
        return self.chain.stream({
            "input": input,
            "chat_history": chat_history
        })

# Create a single instance of the agent to be used by the UI
mdl_agent = MdlAgent()