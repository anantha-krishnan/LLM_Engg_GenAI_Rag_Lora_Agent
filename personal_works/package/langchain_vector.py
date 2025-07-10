import os
import ast, json, re
from dotenv import load_dotenv

# --- Imports from LangChain ---
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
# IMPORT THE LANGUAGE ENUM FOR THE CODE-AWARE SPLITTER
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_community.vectorstores import FAISS
from langchain.tools import tool
from langchain.prompts import PromptTemplate
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor

import global_vars
# ==============================================================================
# SECTION 1: FOUNDATION - SETUP RAG RETRIEVER
# ==============================================================================
load_dotenv()
REPO_PATH = 'C:/Projects/llm_engg/llm_engineering/personal_works/' # Search from the current directory where all files are

print("Loading documents from the repository...")
loader = DirectoryLoader(REPO_PATH, glob="**/*.py", show_progress=True, recursive=True)
docs = loader.load()

# --- THIS IS THE CRITICAL FIX ---
# Use a code-aware splitter that understands Python's structure.
print("Splitting documents with a Python-aware splitter...")
text_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=1500, chunk_overlap=200
)
splits = text_splitter.split_documents(docs)

print("Creating vector store...")
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

print(f"Loaded {len(docs)} documents and created retriever.")
print("-" * 50)


# ==============================================================================
# SECTION 2: DEFINE THE AGENT'S TOOLS (The robust workflow)
# ==============================================================================

class CodeFinder(ast.NodeVisitor):
    def __init__(self, target: str):
        if '.' in target:
            self.class_name, self.method_name = target.split('.', 1)
        else:
            self.class_name = None
            self.method_name = target
        self.target_node = None

    def visit_ClassDef(self, node):
        if self.class_name and node.name == self.class_name:
            for body_item in node.body:
                if isinstance(body_item, ast.FunctionDef) and body_item.name == self.method_name:
                    self.target_node = body_item
                    return
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if not self.class_name and node.name == self.method_name:
            self.target_node = node
        self.generic_visit(node)

# --- Tool 1: Find Potential Targets ---
@tool
def find_potential_code_targets(description: str) -> list[dict]:
    """
    Use this tool FIRST to find potential code targets based on a natural language description.
    It performs a semantic search and returns a simple list of file paths and likely target names.

    Args:
        description (str): A detailed description of the function or method's purpose.
                           For example: "a function that adds a new tool to the Gemini agent".

    Returns:
        list[dict]: A list of dictionaries, each with 'file_path' and a likely 'target_name'
                    (e.g., 'ClassName.method_name' or 'function_name').
    """
    print(f"--- Executing find_potential_code_targets for: '{description}' ---")
    retrieved_docs = retriever.get_relevant_documents(description)
    if not retrieved_docs:
        return [{"error": "No relevant code found for that description."}]

    llm = OpenAI(temperature=0,model=global_vars.model_openai_4omini)
    # llm = OpenAI(temperature=0)
    targets = []
    processed_targets = set()

    for doc in retrieved_docs:
        file_path = doc.metadata.get('source')
        snippet_code = doc.page_content
        # The "Best" Few-Shot prompt
        prompt = f"""
        You are an expert code analyst. Your task is to identify the full name of a function or method from a snippet of Python code.
        Your response MUST be a single JSON object with the key "function_name".

        ---
        EXAMPLE 1
        Snippet:
        ```python
        class MyAgent:
            def process_data(self, data):
                return self._helper(data)
        ```
        Response:
        {{"function_name": "MyAgent.process_data"}}
        ---
        EXAMPLE 2
        Snippet:
        ```python
        import os
        def load_from_file(path):
            with open(path, 'r') as f:
                return f.read()
        ```
        Response:
        {{"function_name": "load_from_file"}}
        ---
        EXAMPLE 3
        Snippet:
        ```python
            x = y + 1
            if x > 10:
                print("done")
        ```
        Response:
        {{"function_name": "N/A"}}
        ---
        REAL TASK
        Snippet:
        ```python
        {snippet_code}
        ```
        Response:
        """
        cleaned_target_name=None
        raw_response = llm.invoke(prompt).strip()
        try:
            # Clean up potential markdown formatting from the LLM
            raw_response=raw_response.replace(' ','')

            # The improved pattern to ensure matching quotes and extract clean values
            improved_pattern = r'([\'"`])(.*?)\1\s*:\s*([\'"`])(.*?)\3'
            match = re.search(improved_pattern, raw_response)

            if match:
                if match.group(2)=='function_name':
                    cleaned_target_name = match.group(4)
        except Exception as e:
            print(f"    [Tool Warning] Failed to parse literal response: {raw_response}. Error: {e}")
            cleaned_target_name = None

        if cleaned_target_name and cleaned_target_name != "N/A":
            target_key = f"{file_path}|{cleaned_target_name}"
            if target_key not in processed_targets:
                print(f"    [Tool] Identified potential target: {cleaned_target_name} in {file_path}")
                targets.append({"file_path": file_path, "target_name": cleaned_target_name})
                processed_targets.add(target_key)
            
    targets = targets if targets else [{"error": "Could not identify any specific function/method names from the search results."}]
    del llm
    return targets

def get_code_by_name(file_path: str, target_name:str) -> dict:
    """
    Use this tool AFTER `find_potential_code_targets` to get the full source code.
    It takes a string representation of a dictionary containing a file path and target name,
    and uses an AST parser to extract the code precisely.

    Args:
        tool_input_string (str): A string that looks like a Python dictionary with two keys:
                                 'file_path' (str): The full path to the Python file.
                                 'target_name' (str): The specific name (e.g., 'GeminiAgent.add_tool').
                                 Example: "{'file_path': 'path/to/file.py', 'target_name': 'MyClass.my_method'}"
    """
    
    if not file_path or not target_name:
        return {"error": "The input must contain both 'file_path' and 'target_name' keys."}

    print(f"--- Executing get_code_by_name for: {target_name} in {file_path} ---")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source_code_full = f.read()
        
        tree = ast.parse(source_code_full)
        finder = CodeFinder(target_name)
        finder.visit(tree)
        
        target_node = finder.target_node
        if target_node:
            function_code = ast.get_source_segment(source_code_full, target_node)
            return {"file_path": file_path, "source_code": function_code}
        else:
            return {"error": f"Could not find '{target_name}' in the AST of {file_path}."}
            
    except Exception as e:
        return {"error": f"An error occurred during extraction: {e}"}


# --- Tool 3 : Refactoring ---
@tool
def refactor_code(tool_input_string: str) -> str:
    """
    Use this tool AFTER `find_potential_code_targets` refactor the source code.
    It takes a string representation of a dictionary containing a file path and target name,
    and uses an AST parser to extract the code precisely.

    Args:
        tool_input_string (str): A string that looks like a Python dictionary with three keys:
                                 'file_path' (str): The full path to the Python file.
                                 'target_name' (str): The specific name (e.g., 'GeminiAgent.add_tool').
                                 'modification_request (str): The message from user explaining the modification request
                                 Example: "{'file_path': 'path/to/file.py', 'target_name': 'MyClass.my_method', 'modification_request': 'Refactor the code to improve efficiency'}"
    Returns:
        str: The new, improved source code as a single string.
    """
    tool_input = ast.literal_eval(tool_input_string)
    file_path = tool_input.get('file_path')
    target_name = tool_input.get('target_name')
    original_code = get_code_by_name(file_path=file_path, target_name=target_name)
    modification_request = tool_input.get('modification_request')

    llm = OpenAI(temperature=0, model=global_vars.model_openai_4omini)
    # llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo-instruct")
    prompt = f"""You are an expert Python programmer. Modify the following code based on the user's request.
    Return ONLY the complete, new Python code. Do not add any commentary.

    Original Code:
    ```python
    {original_code.get('source_code')}
    ```

    Modification Request:
    "{modification_request}"
    """
    output = llm.invoke(prompt)
    del llm
    return output


def save_code_to_file(file_path: str, new_code: str, original_code: str) -> str:
    """Saves a new block of code to a file, replacing an original block."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        if original_code not in content:
            return f"Error: Original code block not found in {file_path}. Cannot save changes."
        new_content = content.replace(original_code, new_code)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return f"Success: New code saved to {file_path}"
    except Exception as e:
        return f"Error writing to file: {e}"

# --- Agent Creation ---
tools = [
    find_potential_code_targets,    
    refactor_code,     
]
react_prompt = hub.pull("hwchase17/react")
prompt_template = react_prompt.template
new_instruction = """
IMPORTANT: You are an advanced AI assistant that can read and modify a local Python codebase.

Your primary purpose is to follow a sequence of steps to fulfill a user's code modification request:
1.  **Find Targets:** Use the `find_potential_code_targets` tool with a descriptive query to locate potential targets.
2.  **Analyze & Decide:** If the tool returns a list, you MUST analyze it. State which item you are choosing and why, based on the user's request.
3.  **Modify:** Use `refactor_code` to make the requested changes.

If a tool returns an error, you must stop and report the error to the user. Do not try to guess or hallucinate.
If the user's request does not require tools (e.g., a general knowledge question), then respond directly.
"""
hybrid_prompt = PromptTemplate.from_template(new_instruction + prompt_template)
llm = OpenAI(temperature=0,  model=global_vars.model_openai_4omini)
hybrid_agent = create_react_agent(llm, tools, hybrid_prompt)
agent_executor = AgentExecutor(
    agent=hybrid_agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=15
)

print("\n--- Testing Semantic Search and Code Modification ---")
task = """
Find the function in the Gemini agent file that is responsible for adding a new tool to the list of available tools.
Then, modify it to print the name of the function that is being added.
"""
result = agent_executor.invoke({"input": task})
del llm
print("\nFinal Answer:", result['output'])
print("-" * 50)