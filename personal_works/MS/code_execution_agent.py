# File: code_execution_agent.py

import ast
import json
import re
from textwrap import dedent
import asyncio
import websockets

from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import tool
from langchain.prompts import PromptTemplate
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from typing import List, Any, Tuple, Dict

import global_vars # Import our shared variables

def validate_user_function(function_code: str) -> Tuple[bool, str, ast.AST]:
    """
    Inspects the user's Python code using AST to enforce rules.
    Rules:
    1. Must be a single, valid function.
    2. All arguments must have type hints.
    3. The function's return type hint must be `dict` or `Dict`.
    """
    try:
        # Clean up potential markdown code blocks from the LLM
        #cleaned_code = re.sub(r'^```python\n|```$', '', function_code, flags=re.MULTILINE).strip()
        
        cleaned_code = dedent(function_code.strip())
        tree = ast.parse(cleaned_code)

        # Rule 1: Must be a single function
        if not tree.body or not isinstance(tree.body[-1], ast.FunctionDef):
            return False, "Error: The code must be a single, valid Python function.", tree
        
        func_def = tree.body[-1]

        # Rule 2: All arguments must have type hints
        for arg in func_def.args.args:
            if arg.annotation is None:
                return False, f"Error: Argument '{arg.arg}' in function '{func_def.name}' is missing a type hint.", tree

        # Rule 3: Return type hint must be dict or Dict
        if func_def.returns is None:
            return False, f"Error: Function '{func_def.name}' is missing a return type hint. It must be '-> dict:'.", tree

        # ast.unparse is the modern way to get the annotation as a string
        return_type_str = ast.unparse(func_def.returns)
        if return_type_str not in ['dict', 'Dict']:
            return False, f"Error: Function '{func_def.name}' must have a return type hint of 'dict' or 'Dict', but found '{return_type_str}'.", tree

    except SyntaxError as e:
        return False, f"Syntax Error: The provided code is not valid Python. Details: {e}", None
    except Exception as e:
        return False, f"An unexpected error occurred during validation: {e}", None
    
    return True, "Validation successful.",tree

# --- Toolset for the Specialist Agent ---

# Global dictionary to store user-defined functions
USER_FUNCTIONS = {}
HELPER_FUNCTIONS = {}

# Original docstring for the run tool, used as a template
RUN_TOOL_ORIGINAL_DOCSTRING = "Runs a previously registered user function on the user's machine via a WebSocket."

def _update_run_tool_docstring():
    """Dynamically updates the 'run' tool's docstring with available functions."""
    # This helps the LLM know which functions it can call.
    func_signatures = [
        f"- {name}(...): {info['description'][:60].strip()}..."
        for name, info in USER_FUNCTIONS.items()
    ]
    available_funcs_str = "\n".join(func_signatures) or "No functions have been registered yet."
    new_docstring = RUN_TOOL_ORIGINAL_DOCSTRING + "\n\nAvailable functions:\n" + available_funcs_str
    # Directly modify the __doc__ of the tool function
    run_user_function_via_websocket.__doc__ = new_docstring

@tool
def run_user_function_via_websocket(json_input: str) -> str:
    """
    Runs a registered user function via a websocket. The entire input to this tool MUST be a single string containing a valid JSON object.

    The JSON object within the string MUST contain two keys:
    1. "function_name": The exact name of the function to run.
    2. "kwargs": A dictionary of the arguments to pass to the function.
    """
    try:
        # Clean the input string which might come with extra quotes from the LLM
        clean_json_str = json_input.strip().strip("'\"")
        data = json.loads(clean_json_str)
        
        function_name = data.get("function_name")
        kwargs = data.get("kwargs")

        if not function_name or not isinstance(kwargs, dict):
            return "Error: The parsed JSON is missing required keys. It must contain 'function_name' (string) and 'kwargs' (dictionary)."

    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON format in input string. Please provide a valid JSON string. Details: {e}"
    except Exception as e:
        return f"Error: Failed to parse input. Expected a JSON string. Details: {e}"
    
    if function_name not in USER_FUNCTIONS:
        return f"Error: Function '{function_name}' is not registered. Available functions are: {list(USER_FUNCTIONS.keys())}"

    function_info = USER_FUNCTIONS[function_name]
    
    # --- WebSocket Communication ---
    async def _run_job_async():
        uri = "ws://localhost:8765/run_job"
        payload = {"function_code": function_info["code"], "kwargs": kwargs, "helper_functions": [hf["code"] for hf in HELPER_FUNCTIONS.values()]}
        try:
            async with websockets.connect(uri) as websocket:
                await websocket.send(json.dumps(payload))
                response = await websocket.recv()
                return json.loads(response)
        except Exception as e:
            return {"status": "error", "message": f"Failed to connect or communicate with the WebSocket bridge server: {e}"}

    # Ensure we run asyncio correctly
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    result = loop.run_until_complete(_run_job_async())
    
    return f"Execution Result: {result}"


class RegisterToolInput(BaseModel):
    function_code: str = Field(description="A string containing the complete Python function, including the 'def' signature, docstring, and body.")

def unparse_code(tree: ast.AST) -> str:
    """Converts an AST back to source code."""
    try:
        func_def = tree.body[-1]
        func_name = func_def.name
        docstring = ast.get_docstring(func_def) or "No docstring provided."
        # get the cleaned code again
        cleaned_code = ast.unparse(tree)    
        # Store the function details
        return cleaned_code, func_name, docstring
    except Exception as e:
        raise ValueError(f"Could not unparse AST to code. Details: {e}")
    
def save_helper_function(tree: ast.AST) -> str:
    """
    Saves a helper function to a local file named 'helper_functions.py'.
    If the file already exists, it appends the new function.
    """
    try:
        cleaned_code, func_name, docstring = unparse_code(tree)
        HELPER_FUNCTIONS[func_name] = {
            "code": cleaned_code, 
        }
        return "Success: Helper function saved to 'helper_functions.py'."
    except Exception as e:
        return f"Error: Could not save helper function. Details: {e}"
    
# @tool("register_user_function", args_schema=RegisterToolInput)
def register_user_function(tree: ast.AST) -> str:
    """Parses and registers a new Python function from a multi-line string to make it available for execution."""
    try:
        cleaned_code, func_name, docstring = unparse_code(tree)
        # Store the function details
        USER_FUNCTIONS[func_name] = {
            "code": cleaned_code, 
            "description": docstring
        }
        
        # Update the run tool's help text so the agent knows about the new function
        _update_run_tool_docstring()
        
        return f"Success: Function '{func_name}' was registered. It is now available to the 'run_user_function_via_websocket' tool."
    except Exception as e:
        return f"Error: An unexpected error occurred while parsing the function code. Details: {e}"

# --- Agent Factory Function ---

def create_code_executor() -> AgentExecutor:
    """
    Factory function to create the specialized Code Execution Agent.
    This agent is a "tool user" that doesn't reason about high-level goals.
    """
    tools = [run_user_function_via_websocket]
    react_prompt = hub.pull("hwchase17/react")

    instructions = """
    You are a specialized AI assistant that executes Python code on a user's machine. You do not write code, you only execute it using your tools. Your process is very strict:

    1.  **REGISTER**: When a user provides a Python function, you MUST use the `register_user_function` tool. Pass the entire, multi-line code block into the `function_code` argument.

    2.  **RUN**: When asked to run a function, you MUST use the `run_user_function_via_websocket` tool. The input for this tool MUST BE A SINGLE STRING containing a valid JSON object. The keys in the `kwargs` dictionary inside the JSON string MUST EXACTLY MATCH the argument names of the function.
    """
    
    prompt = react_prompt.partial(instructions=instructions)
    
    # Use the faster, cheaper model for this simple, rule-based agent
    llm = ChatOpenAI(temperature=0, model=global_vars.model_openai_4omini)
    
    agent = create_react_agent(llm, tools, prompt)
    
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        max_iterations=10, 
        verbose=True, # Set to False for cleaner production logs
        handle_parsing_errors="""
        Could not parse LLM output. Please make sure to provide a valid Action and Action Input.
        For the `run_user_function_via_websocket` tool, the Action Input MUST be a single string containing a valid JSON object.
        Example: '{"function_name": "my_func", "kwargs": {"arg1": "value1"}}'
        Do not add any other text outside this string.
        """
    )
    return agent_executor