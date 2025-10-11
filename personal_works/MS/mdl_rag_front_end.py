import re
import os
import re
from typing import List, Dict, Any
import pandas as pd
from pathlib import Path

# LangChain components for the knowledge base
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
import gradio as gr
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.chat_models import ChatOpenAI # Use ChatOpenAI for self-query
from langchain.schema import BaseMessage
from dataclasses import dataclass

## llm imports
import openai
import anthropic

#local imports
import global_vars

# Initialize OpenAI client
client = openai.OpenAI()
# Initialize Anthropic client
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
MDL_LIBRARY_PATH = "C:/Altair_Installs/2026_0_10_release/hwdesktop/hw/mdl/mdllib/Libs" # MAKE SURE YOU CREATE THIS FOLDER AND ADD .mdl FILES
# --- Vector Store Management ---
DB_DIR = "./chroma_db_mdl"

#hwdesktop\hw\mdl\mdllib\Libs\Models\Frnt_susp
 
#hwdesktop\hw\mdl\mdllib\Libs\Models\Steering\Linkages
 
#t = mview.System (name = 'sys_steering', label = "Rack and Pinion", definition_file = "C:/Altair_Installs/2026.0.0.17/hwdesktop/hw/mdl/mdllib/Libs/Models/Steering/Linkages/Rackpin/rackpin.mdl",  definition_name = "sysdef_str_links")
@dataclass
class MdlComponent:
    file_path: str
    system_def_name: str
    description: str
    attachment_candidates: List[str]
    file_name: str
    # Additional metadata fields from CSV
    model_type: str = ""
    side: str = ""
    sub_category_1: str = ""
    sub_category_2: str = ""

            
def get_self_query_retriever(db_dir: str = DB_DIR)->SelfQueryRetriever:
    """
    Creates and returns a SelfQueryRetriever that can parse natural language
    into structured metadata filters.
    Args:
        db_dir (str): Directory where the Chroma database is stored.
    Returns:
        SelfQueryRetriever: The configured retriever.
    """
    if not os.path.exists(db_dir):
        raise FileNotFoundError("Vector store not found. Please build it first.")

    # 1. Define the metadata fields the LLM can filter on
    metadata_field_info = [
        AttributeInfo(
            name="model_type",
            description="The main type of the component, such as 'Suspension', 'Driveline', 'Bumper', 'Body'.",
            type="string",
        ),
        AttributeInfo(
            name="side",
            description="The position of the component on the vehicle, such as 'Front', 'Rear'. Use 'NA' if not applicable.",
            type="string",
        ),
        AttributeInfo(
            name="category1",
            description="The primary sub-category of the component. For example, for a Suspension, this could be 'Macpherson', 'Multilink', or 'SLA'.",
            type="string",
        ),
        AttributeInfo(
            name="category2",
            description="The secondary sub-category, providing more detail. For example, for a Macpherson suspension, this could be '1 piece' or '2 piece'.",
            type="string",
        ),
    ]
    
    # 2. Load the vector store
    vector_store = Chroma(persist_directory=db_dir, embedding_function=OpenAIEmbeddings())
    
    # 3. Instantiate the LLM and Retriever
    # Self-query requires a chat model that supports function calling
    llm = ChatOpenAI(model=global_vars.model_openai_4omini, temperature=0) 
    
    document_content_description = "Brief description of a vehicle component from a modeling library"
    
    retriever = SelfQueryRetriever.from_llm(
        llm,
        vector_store,
        document_content_description,
        metadata_field_info,
        verbose=True # Set to True to see the generated queries!
    )
    
    return retriever
self_query_retriever = get_self_query_retriever()

def get_def_sys_arg(content: str) -> Dict[str, str]:
    """
    Extracts the arguments from a *DefineSystem command statement in a .mdl file.
    Args:
        content (str): The content of the .mdl file.
    Returns:
        Dict[str, Any]: A dictionary with keys 'system_name' and 'attachment_candidates'.
    """
    pattern = re.compile(
        r"\*DefineSystem\("         # Match the literal start of the function call
        r"\s*(\w+)\s*,"             # Group 1: Capture the first argument (one or more word characters)
        r"(.*?)"                    # Group 2: Non-greedily capture everything else...
        r"\)",                      # ...until we hit the closing parenthesis
        re.DOTALL | re.IGNORECASE   # Use DOTALL for multi-line matching, IGNORECASE for robustness
    )

    match = pattern.search(content)
    ret_value = {'system_name': None, 'attachment_candidates': []}
    if match:
        # The first argument is in group 1
        first_argument = match.group(1).strip()
        
        # The rest of the arguments are in a single string in group 2        
        raw_other_args = match.group(2)
        other_arguments = [arg.strip() for arg in raw_other_args.split(',') if arg.strip()]

        #print(f"Full matched block:\n---\n{match.group(0)}\n---\n")
        #print(f"First Argument: {first_argument}")
        #print(f"Other Arguments: {other_arguments}")

        ret_value['system_name'] = first_argument
        ret_value['attachment_candidates'] = other_arguments
    else:
        print("DefineSystem block not found.")
        pass

    return ret_value
        
def get_mdl_csv_metadata(file_path: str) -> pd.DataFrame:
    pd_df = pd.read_csv(file_path)
    pd_df.fillna("", inplace=True)
    # join columns file_name and folder to create full path
    pd_df['folder'] = pd_df['folder'].apply(lambda x: Path(x).as_posix())
    pd_df['full_path'] = pd_df['folder'] + '/' + pd_df['file_name']
    pd_df['full_path'] = pd_df['full_path'].apply(lambda x: Path(x).as_posix())  # Ensure consistent path format
    return pd_df

def update_mdl_components_with_csv_data(components: List[MdlComponent], csv_path: str) -> List[MdlComponent]:
    """
    Updates the list of MdlComponent objects with additional metadata from a CSV file.
    Args:
        components (List[MdlComponent]): List of MdlComponent objects to update.
        csv_path (str): Path to the CSV file containing additional metadata.
    Returns:
        List[MdlComponent]: Updated list of MdlComponent objects.
    """
    if not os.path.exists(csv_path):
        print(f"CSV file '{csv_path}' not found. Skipping metadata update.")
        return components

    metadata_df = get_mdl_csv_metadata(csv_path)
    metadata_dict = metadata_df.set_index('full_path').to_dict(orient='index')

    for component in components:
        if component.file_path in metadata_dict:
            meta = metadata_dict[component.file_path]
            # Assuming MdlComponent has attributes matching the CSV columns
            for key, value in meta.items():
                if hasattr(component, key) and value:
                    setattr(component, key, value)
    
    return components

def get_mdl_component(file_path)->MdlComponent:
    """
    Parses a .mdl file to extract key information for LLM processing.
    Args:
        file_path (str): Path to the .mdl file.
    Returns:
        str: A summary of the file's key information.
    """
    with open(file_path, 'r') as f:
        content = f.read()
    file_path = Path(file_path).as_posix()  # Ensure consistent path format
    # Extract description from comments at the top
    description = None
    for line in content.splitlines():
        if line.startswith('//'):
            if 'Description:' in line:
                description = line.split('Description:')[1].strip()
        else:
            # Stop after the first non-comment line
            if description: break
    if not description:
        #print(f"No description found in {file_path}. Skipping.")
        return None
    # Extract system definition name and attachment candidates
    def_sys_args = get_def_sys_arg(content)
    system_name = def_sys_args.get('system_name', "Unknown")
    attachment_candidates = def_sys_args.get('attachment_candidates', [])
    
    # Combine into a single document for the LLM
    component = MdlComponent(
        file_path=file_path,
        system_def_name=system_name,
        description=description.strip(),
        attachment_candidates=attachment_candidates,
        file_name=file_path.split('/')[-1]
    )
    return component
    summary = f"""
    Component File: {file_path.split('/')[-1]}
    Friendly Name: {friendly_name}
    System Definition Name: {system_name}
    Description: {description}
    Attachment Candidates: {', '.join(attachment_candidates) if attachment_candidates else 'None'}
    """
    return summary

def get_valid_mdl_components(mdl_library_path: str) -> List[MdlComponent]:
    """
    Scans the specified directory for .mdl files and returns a list of valid files
    that contain a description.
    Args:
        mdl_library_path (str): Path to the directory containing .mdl files.
    Returns:
        List[MdlComponent]: A list of valid .mdl components.
    """
    valid_mdl_components = []
    invalid_files = []
    # Process each .mdl file in the library path
    for root, dirs, files in os.walk(mdl_library_path):
        for file in files:
            if file.endswith(".mdl"):
                file_path = os.path.join(root, file)
                component = get_mdl_component(file_path)
                if component:
                    if component.description:
                        valid_mdl_components.append(component)
                    else:
                        invalid_files.append(file)
                else:
                    invalid_files.append(file)
    if invalid_files:
        print(f"Skipped {len(invalid_files)} files without descriptions:")
        #for f in invalid_files:
        #    print(f" - {f}")
    return valid_mdl_components

def build_mdl_vector_store(mdl_library_path: str = MDL_LIBRARY_PATH, db_dir: str = DB_DIR, force_rebuild: bool = False, csv_path: str = None):
    """
    Builds a Chroma vector store from .mdl files in the specified directory.
    Args:
        mdl_library_path (str): Path to the directory containing .mdl files.
        db_dir (str): Directory to store the Chroma database.
    Returns:
        Chroma: The populated Chroma vector store.
    """
    if os.path.exists(DB_DIR) and not force_rebuild:
        print("Vector store already exists. Loading...")
        return

    print("Building vector store...")
    
    documents=[]
    components=get_valid_mdl_components(mdl_library_path)
    components = update_mdl_components_with_csv_data(components, csv_path)
    print(f"Found {len(components)} valid .mdl files with descriptions.")
    for component in components:
        # Add component to vector store
        # The page_content is what gets searched, metadata is what we retrieve
        page_content = (
            f"Component: {component.sub_category_1} {component.sub_category_2}. "
            f"Type: {component.model_type}. Side: {component.side}. "
            f"Description: {component.description}"
        )

        # --- Create the rich metadata for filtering ---
        metadata = {
            "file_name": component.file_name,
            "file_path": component.file_path,
            "model_type": component.model_type,
            "side": component.side,
            "category1": component.sub_category_1,
            "category2": component.sub_category_2,
            "system_def_name": component.system_def_name,
            "attachment_candidates": ", ".join(component.attachment_candidates),
        }
        documents.append(
            Document(
                page_content=page_content,
                metadata=metadata
            )
        )
    if documents:
        # Initialize embeddings and create the vector store
        embeddings = OpenAIEmbeddings()
        vector_store = Chroma.from_documents(documents, embeddings, persist_directory=db_dir)
        print(f"Added {len(documents)} documents to the vector store.")
                

    # Persist the vector store
    vector_store.persist()



def format_context(matches: List[Dict[str, Any]]) -> str:
    """Formats the retrieved documents into a string for the LLM prompt."""
    if not matches:
        return "No relevant components found in the library."
    
    context_str = "Here are some relevant components found in the library:\n\n"
    for i, match in enumerate(matches):
        meta = match['metadata']
        context_str += f"--- Component {i+1} ---\n"
        # context_str += f"Content: {match['content']}\n"
        context_str += f"File Path: {meta.get('file_path', 'N/A')}\n"
        context_str += f"System Definition: {meta.get('system_def_name', 'N/A')}\n"
        context_str += f"Attachment Candidates: {meta.get('attachment_candidates', 'N/A')}\n\n"
    return context_str

def find_vehicle_component(query: str, db_dir: str = DB_DIR, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Searches the Chroma vector store for components matching the query.
    Args:
        query (str): The search query.
        db_dir (str): Directory where the Chroma database is stored.
        top_k (int): Number of top results to return.
    Returns:
        List[Dict[str, Any]]: A list of matching components with metadata.
    """
    if not os.path.exists(db_dir):
        raise FileNotFoundError("Vector store not found. Please build it first.")

    # Load the existing vector store
    vector_store = Chroma(persist_directory=db_dir, embedding_function=OpenAIEmbeddings())
    
    # Perform the similarity search
    print(f"Searching for components matching query: '{query}'")
    results = vector_store.similarity_search_with_score(query, k=top_k)
    print(f"Found {len(results)} matching components.")
    # Extract relevant information from results
    matches = []
    for doc, score in results:
        matches.append({
            "description": doc.page_content,
            "metadata": doc.metadata,
            "similarity_score": score
        })
    
    return matches

def get_intent(message: str, chat_history: List) -> str:
    """
    Uses an LLM to determine the user's intent.
    Args:
        message (str): The latest user message.
        chat_history (List[BaseMessage]): The conversation history.
    Returns:
        str: The detected intent ('find_component', 'select_component', or 'general_conversation').
    """

    system_prompt = """
    You are an intent classifier for a CAE software assistant. Your job is to determine the user's primary intent based on their latest message and the conversation history.
    There are only three possible intents:
    1. 'find_component': The user wants to search for a component. This is for initial queries or new searches.
        Examples: "add a suspension", "find a shock absorber", "what steering systems do you have?"

    2. 'select_component': The user is referring to or choosing a component from a list you have ALREADY presented in the conversation. This is often a follow-up action.
        Examples: "get me component 2", "I'll take the first one", "select the Macpherson suspension", "give me more details on that one"

    3. 'general_conversation': The user is asking a general question, greeting you, or having a conversation that is NOT about finding or selecting a specific component.
        Examples: "hello", "what is MotionView?", "can you help me?"
    
    Look at the last assistant message. If it was a list of components, and the user is now referring to one of them by number or name, the intent is 'select_component'.

    Respond with ONLY the word 'find_component', 'select_component', or 'general_conversation'.
    """
    messages = [
        {"role": "system", "content": system_prompt},
    ]
    # Add history for context
    if chat_history:
        for user_msg, bot_msg in chat_history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": bot_msg})
    
    messages.append({"role": "user", "content": message})

    query_gen_response = client.chat.completions.create(
        messages=messages,
        model=global_vars.model_openai_4omini, # Or another fast model
        temperature=0
    )
    intent = query_gen_response.choices[0].message.content.strip()
    print(f"Intent: '{intent}'")

    # Simple validation
    if "find_component" in intent:
        return "find_component"
    if "select_component" in intent:
        return "select_component"
    return "general_conversation"

def get_general_response(message: str, chat_history: List) -> str:
    """
    Gets a conversational response from the LLM with motion view as context.
    Args:
        message (str): The latest user message.
        chat_history (List[BaseMessage]): The conversation history.
    Returns:
        str: The LLM's response.
    """
    system_prompt = "You are a helpful assistant for the Altair MotionView software. Be friendly and concise."
    messages = [
        {"role": "system", "content": system_prompt},
    ]
    for user_msg, bot_msg in chat_history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": message})
    response = client.chat.completions.create(
        messages=messages,
        model=global_vars.model_openai_4omini, 
        temperature=1.0,
        stream=True
    )
    partial_response = ""
    for chunk in response:
        if chunk.choices[0].delta.content:
            partial_response += chunk.choices[0].delta.content
            yield partial_response



def chat_with_rag(message:str, history:list)->str:
    """
    The main function that gradio interface calls to handle user queries.
    It performs retrieval from the vector store, builds a prompt and generates a response using an LLM.
    Args:
        message (str): The user's question.
        history (str): The chat history.
    Returns:
        str: The assistant's response.
    """
    search_quantity = 3  # default value
    # 1: Transform the query using conversation history
    if history: # Only do this if there's a history
        history_str = "\n".join([f"Human: {u}\nAssistant: {a}" for u, a in history])
        
        # A new prompt just for creating a better search query
        query_gen_prompt = f"""
        Given the following conversation history and a follow-up question, rephrase the follow-up question 
        to be a standalone query that can be used to search a vector database for vehicle components.

        Conversation History:
        {history_str}

        Follow-up Question: {message}
        Standalone Query:"""
        
        query_gen_response = client.chat.completions.create(
            messages=[{"role": "user", "content": query_gen_prompt}],
            model=global_vars.model_openai_4omini, # Or another fast model
            temperature=0
        )
        search_query = query_gen_response.choices[0].message.content.strip()
        print(f"Original query: '{message}' | Standalone query: '{search_query}'")
    else:
        search_query = message # For the first message, the query is fine as is
    """
    if history: # Only do this if there's a history
        query_gen_prompt = f\"\"\"
            Given the following conversation history and a follow-up question, find out IF the user has requested for a particular the number of choices of vehicle components,
            if yes, get back the number of choices requested, else return 3 as default.
            If the user has not specified a number, return 3.
            If the user has specified a number, return that number only.
            If the user has specified a range, return the upper limit of the range only.
            If the user has specified a number in words, return the numeric equivalent of that number only.
            If the user has specified all, return 5.

            Conversation History:
            {history_str}

            Follow-up Question: {message}
            Number of results required:\"\"\"
        query_gen_response = client.chat.completions.create(
                messages=[{"role": "user", "content": query_gen_prompt}],
                model=global_vars.model_openai_4omini, # Or another fast model
                temperature=0
            )
        search_quantity = query_gen_response.choices[0].message.content.strip()
    # use regex to extract the number from the response
        match = re.search(r'\d+', search_quantity)
        if match:
            search_quantity = int(match.group())
    """
    retrieved_docs = self_query_retriever.get_relevant_documents(search_query)
    # print(f"Search Query: {search_query}")
    # retrieved_matches = find_vehicle_component(search_query,top_k=search_quantity)
    #print(f"Retrieved {len(retrieved_matches)} matches from vector store.")
    matches = [{"metadata": doc.metadata} for doc in retrieved_docs]
    context_for_llm = format_context(matches)
    print(f"Context for LLM:\n{context_for_llm}")
    # 2. Create the prompt for the LLM
    system_prompt = (
        "You are an expert assistant for a vehicle dynamics modeling software. "
        "Your task is to help users find the correct component from a file library. "
        "Use the provided context (retrieved components that precisely match the user's request) to answer. "
        "If no components were found, state that and ask the user to broaden their search criteria."
    )
    # build the full prompt
    # `history` is provided by Gradio and contains the conversation history
    messages = [
        {"role": "system", "content": system_prompt},
    ]
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})

    # Add the current user message with the context
    user_message_with_context = f"User Query: {message}\n\n--- Retrieved Context ---\n{context_for_llm}"
    messages.append({"role": "user", "content": user_message_with_context})

    # 3. Call the LLM to generate a response
    response = client.chat.completions.create(
        messages=messages,
        stream=True,
        model=global_vars.model_openai_4omini,
        temperature=1.0,
    )
    
    partial_response = ""
    for chunk in response:
        if chunk.choices[0].delta.content:
            partial_response += chunk.choices[0].delta.content
            yield partial_response

def handle_component_selection(message: str, chat_history: List) -> str:
    """
    Handles cases where the user selects a component from a previously provided list.
    This function uses the chat history as context, not the vector store.
    """
    system_prompt = """
    You are an information extraction assistant. The user has selected a component from a list you previously provided in the chat history.
    Your task is to carefully analyze the conversation history and the user's latest message to identify exactly which component they are referring to.

    Once identified, extract and present ONLY the details for that specific component (File Path, System Definition Name, Attachment Candidates) in a clear and structured format.
    
    If you are certain which component they mean, you can start your response with something like "Of course, here are the details for Component X:"
    
    If the user's selection is ambiguous or you cannot determine which component they mean from the history, state that you are unsure and ask them to clarify by providing the full name or file path.
    """
    messages = [
        {"role": "system", "content": system_prompt},
    ]
    for user_msg, bot_msg in chat_history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})

    # Add the current user message
    messages.append({"role": "user", "content": message})

    # Call the LLM to generate a response
    response = client.chat.completions.create(
        messages=messages,
        stream=True,
        model=global_vars.model_openai_4omini,
        temperature=0.2, # Lower temperature for more factual extraction
    )
    
    partial_response = ""
    for chunk in response:
        if chunk.choices[0].delta.content:
            partial_response += chunk.choices[0].delta.content
            yield partial_response

def query_dispatcher(message: str, chat_history: List) -> str:
    """
    Determines the user's intent and routes the query to the appropriate handler.
    Args:
        message (str): The user's question.
        chat_history (List): The chat history.
    Returns:
        str: The assistant's response.
    """
    intent = get_intent(message, chat_history)
    if intent == "find_component":
        yield from chat_with_rag(message, chat_history)
    elif intent == "select_component":
        yield from handle_component_selection(message, chat_history)
    else: # general_conversation
        yield from get_general_response(message, chat_history)

# --- Main execution ---

if __name__ == "__main__":
    if not os.path.exists(MDL_LIBRARY_PATH):
        raise FileNotFoundError(f"MDL library path '{MDL_LIBRARY_PATH}' does not exist. Please set it correctly.")
    # get the current file location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "mdl_structure_metadata.csv")
    build_mdl_vector_store(mdl_library_path=MDL_LIBRARY_PATH, force_rebuild=True, csv_path=csv_path)
    # Create and launch the Gradio interface
    """
    demo = gr.ChatInterface(
        fn=chat_with_rag,
        title="Vehicle Component RAG Chatbot",
        description="Ask me to find vehicle components like 'front suspension' or 'steering system'.",
        examples=["I need a front suspension", "Find a rack and pinion steering system"],
        retry_btn=None,
        undo_btn="Delete Previous",
        clear_btn="Clear",
    )
    """
    with gr.Blocks() as demo:
        gr.Markdown("# Vehicle Dynamics Modeling Assistant")
        gr.Markdown("Ask questions about vehicle components and get suggestions from the library.")
        chatbot = gr.Chatbot()
        msg = gr.Textbox(label="Your Question")
        clear = gr.Button("Clear")

        def respond(message, chat_history):
            bot_response = ""
            for partial in query_dispatcher(message, chat_history):
                bot_response = partial
                yield chat_history + [(message, bot_response)]

        msg.submit(respond, [msg, chatbot], chatbot)
        clear.click(lambda: None, None, chatbot, queue=False)
    demo.launch()





