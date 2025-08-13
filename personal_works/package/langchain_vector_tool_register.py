import ast
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import tool
import asyncio
import websockets
import json
from langchain.prompts import PromptTemplate
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import OpenAI
import re
from textwrap import dedent
import global_vars 

# --- No changes in this section ---
# Store the original docstring to prevent repeated appends
RUN_TOOL_ORIGINAL_DOCSTRING = """
Runs a previously registered user function on the user's machine via a WebSocket.
You MUST register a function with `register_user_function` before you can run it.
The available functions are:
""".strip()

USER_FUNCTIONS = {}

def _update_run_tool_docstring():
    """Rebuilds the docstring for the run tool to show rich signatures of available functions."""
    print("   [Internal] Updating run_user_function_via_websocket docstring...")
    func_signatures = []
    for name, info in USER_FUNCTIONS.items():
        args_str = ", ".join([f"{arg['name']}: {arg['type']}" for arg in info['args']])
        signature = f"- {name}({args_str}): {info['description']}"
        func_signatures.append(signature)
    
    available_funcs_str = "\n".join(func_signatures)
    if not available_funcs_str:
        available_funcs_str = "No functions are currently registered."
        
    new_docstring = RUN_TOOL_ORIGINAL_DOCSTRING + "\n" + available_funcs_str
    run_user_function_via_websocket.__doc__ = new_docstring
    # For debugging, let's see the full new docstring
    print(f"   [Internal] Updated run tool docstring:\n---\n{new_docstring}\n---")

class RunFunctionInput(BaseModel):
    function_name:str = Field(description="The exact name of the registered function to run.")
    kwargs:dict = Field(description="A dictionary of keyword arguments to pass to the function.")

@tool
def run_user_function_via_websocket(json_input: str) -> str:
    """
    Runs a registered user function via a websocket. The entire input to this tool MUST be a single string containing a valid JSON object.

    The JSON object within the string MUST contain two keys:
    1. "function_name": The exact name of the function to run.
    2. "kwargs": A dictionary of the arguments to pass to the function.

    Example of the EXACT Action Input format (a single string):
    '{"function_name": "calculate_sale_price", "kwargs": {"base_price": 250, "discount_percentage": 20}}'
    """
    print(f"--- Executing run_user_function_via_websocket with raw string input: '{json_input}' ---")
    try:
        json_input = json_input.strip().strip("'\"")
        # Manually parse the JSON string
        data = json.loads(json_input)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON format in input string. Please provide a valid JSON string. Details: {e}"
    except Exception as e:
        # Catch other potential errors, e.g., if the input isn't a string at all
        return f"Error: Failed to parse input. Expected a JSON string. Details: {e}"

    # Extract the required keys from the parsed dictionary
    function_name = data.get("function_name")
    kwargs = data.get("kwargs")

    # Validate that the necessary keys exist and have the correct type
    if not function_name or not isinstance(kwargs, dict):
        return "Error: The parsed JSON is missing required keys. It must contain 'function_name' (string) and 'kwargs' (dictionary)."

    # Call the actual implementation logic with the parsed data
    return _run_user_function_implementation(function_name=function_name, kwargs=kwargs)

def _run_user_function_implementation(function_name: str, kwargs: dict) -> str:
    print(f"--- Executing run_user_function_via_websocket for: '{function_name}' ---")
    if function_name not in USER_FUNCTIONS:
        return f"Error: Function '{function_name}' is not registered. Available functions are: {list(USER_FUNCTIONS.keys())}"

    # Use the 'code' key which stores the full function string
    function_info = USER_FUNCTIONS[function_name]
    
    async def _run_job():
        uri = "ws://localhost:8765/run_job"
        # The payload to the server needs the executable code
        payload = {"function_code": function_info["code"], "kwargs": kwargs}
        try:
            async with websockets.connect(uri) as websocket:
                await websocket.send(json.dumps(payload))
                response = await websocket.recv()
                return json.loads(response)
        except Exception as e:
            return {"status": "error", "message": f"Failed to connect or communicate with bridge server: {e}"}

    loop = asyncio.get_event_loop()
    if loop.is_running():
        return "Error: asyncio event loop is already running. Cannot execute function."
    return_res = loop.run_until_complete(_run_job())
    
    print(f"   [Tool] Received response from bridge: {return_res}")
    return f"Execution Result: {return_res}"

# --- End of unchanged section ---



# 1. More explicit Pydantic model for registering a function.
# This guides the LLM to pass the code as a complete, multi-line block.
class RegisterToolInput(BaseModel):
    function_code: str = Field(
        description="A string containing the complete Python function, including the 'def' signature, docstring, and body. It is critical to preserve all original newlines and indentation."
    )


@tool("register_user_function", args_schema=RegisterToolInput)
def register_user_function(function_code: str) -> str:
    """
    Parses and registers a new Python function from a multi-line string to make it available for execution.
    The function code MUST be a complete block starting with 'def', and MUST contain a proper docstring and type hints for all arguments.
    
    Args:
        function_code (str): The complete function code as a multi-line string.
    
    Returns:
        A string message confirming the registration outcome, including the function's exact signature.
    """
    # (No changes to the initial cleaning and parsing logic)
    print(f"   [Internal] Received raw function code block from LLM...")
    cleaned_code = re.sub(r'^```python\n|```$', '', function_code, flags=re.MULTILINE).strip()
    try:
        cleaned_code = dedent(cleaned_code)
    except Exception as e:
        return f"Error: Failed during indentation cleanup. {e}"
    print(f"   [Internal] Final cleaned code for parsing: \n---\n{cleaned_code}\n---")
    try:
        tree = ast.parse(cleaned_code)
        if not tree.body or not isinstance(tree.body[0], ast.FunctionDef):
            return "Error: The provided code does not appear to be a single, valid Python function."
        func_def = tree.body[0]
        func_name = func_def.name
        docstring = ast.get_docstring(func_def)
        if not docstring:
            return f"Error: Function '{func_name}' is missing a docstring. Registration requires a docstring."
        arg_info = []
        for arg in func_def.args.args:
            if not arg.annotation:
                return f"Error: Argument '{arg.arg}' in function '{func_name}' is missing a type hint. Registration requires all arguments to have type hints."
            arg_info.append({
                "name": arg.arg,
                "type": ast.unparse(arg.annotation) if hasattr(ast, 'unparse') else arg.annotation.id
            })
            
        USER_FUNCTIONS[func_name] = {
            "code": cleaned_code,
            "description": docstring,
            "args": arg_info
        }
        _update_run_tool_docstring()
        
        # --- THIS IS THE KEY CHANGE ---
        # Create the full signature string to return to the agent.
        args_str = ", ".join([f"{arg['name']}: {arg['type']}" for arg in arg_info])
        full_signature = f"{func_name}({args_str})"
        
        # Return a highly informative message that the agent can use in the next step.
        return f"Success: Function '{func_name}' was registered. You can now run it using its exact signature: {full_signature}"

    except SyntaxError as e:
        return f"Error: The provided code has a syntax error. Please fix the code. Details: {e}"
    except Exception as e:
        return f"Error: An unexpected error occurred while parsing the code. Details: {e}"

class GetSignatureInput(BaseModel):
    function_name: str = Field(description="The name of the function to inspect.")

@tool("get_function_signature", args_schema=GetSignatureInput)
def get_function_signature(function_name: str) -> str:
    """
    Gets the exact signature (name and arguments) for a previously registered function.
    This MUST be called before using the 'run_user_function_via_websocket' tool.
    """
    print(f"--- Executing get_function_signature for: '{function_name}' ---")
    if function_name not in USER_FUNCTIONS:
        return f"Error: Function '{function_name}' is not registered."

    info = USER_FUNCTIONS[function_name]
    args_str = ", ".join([f"{arg['name']}: {arg['type']}" for arg in info['args']])
    full_signature = f"{info['code'].splitlines()[0].strip().replace(':', '')}" # Get the def line
    
    # Return a message that is both human-readable and easy for the LLM to parse.
    return (
        f"Signature for '{function_name}':\n"
        f"  Function: {full_signature}\n"
        f"  Arguments for kwargs: {info['args']}\n"
        f"You MUST use these argument names as keys in the 'kwargs' dictionary for the run tool."
    )
# Link the implementation function
# run_user_function_via_websocket.func = _run_user_function_implementation

tools = [register_user_function, get_function_signature, run_user_function_via_websocket]
react_prompt = hub.pull("hwchase17/react")

### MODIFIED ###
# 2. Strengthened instructions to be more direct.
new_instruction = """
You are a specialized AI assistant that can execute Python code on a user's machine.

Your process is very strict:
1.  **REGISTER**: When the user provides a Python function, you MUST use the `register_user_function` tool. Pass the entire, multi-line code block into the `function_code` argument. retain the line breaks and indentation as it is.
2.  **GET SIGNATURE**: After registering a function, or if you need to run an existing function, you MUST first use the `get_function_signature` tool to find its exact argument names. It is used to form a string input in the form of a well informed JSON
3.  **RUN**: After you have the signature, use the `run_user_function_via_websocket` tool. The input for this tool MUST BE A SINGLE STRING containing a valid JSON object. The keys in the `kwargs` dictionary inside the JSON string MUST EXACTLY MATCH the argument names from the signature.

    **Example of the EXACT Action Input format (as a single string):**
    ```
    '{"function_name": "the_name_from_step_2", "kwargs": {"arg1_from_step_2": "value1", "arg2_from_step_2": "value2"}}'
    ```
    Do not deviate from this single-string JSON format.
If the user asks to perform a task that a registered function can do, you MUST use that function.
"""
hybrid_prompt = PromptTemplate.from_template(react_prompt.template).partial(instructions=new_instruction)
# llm = OpenAI(temperature=0, model=global_vars.model_openai_4omini) # Your LLM init
llm = OpenAI(temperature=0, model="gpt-4o-mini") # Using a placeholder for demonstration
hybrid_agent = create_react_agent(llm, tools, hybrid_prompt)
agent_executor = AgentExecutor(agent=hybrid_agent, tools=tools, max_iterations=10, verbose=True, handle_parsing_errors=True)

if __name__ == "__main__":
    print("\n--- Agent is ready ---")
    print("First, I will register a well-formed function. Then I will ask the agent to use it.")

    # --- Task 1: Register the function ---
    # The input prompt remains the same. The LLM will now know how to correctly
    # package the code block for the `register_user_function` tool.
    task1 = """
    Please register this new function for me. Do not change the indentation and line breaks.

    ```python
    def calculate_sale_price(base_price: float, discount_percentage: float) -> float:
        \"\"\"
        Calculates the final price after applying a discount.
        
        Args:
            base_price: The original price of the item.
            discount_percentage: The discount to apply, as a percentage (e.g., 15 for 15%).
            
        Returns:
            The final price after the discount is applied.
        \"\"\"
        if not 0 <= discount_percentage <= 100:
            raise ValueError("Discount percentage must be between 0 and 100.")
        final_price = base_price * (1 - discount_percentage / 100)
        return round(final_price, 2)
    ```
    """
    result1 = agent_executor.invoke({"input": task1})
    print("\nFinal Answer for Task 1:", result1['output'])
    print("-" * 50)

    # --- Task 2: Use the newly registered function ---
    task2 = "Excellent. Now, use the registered function calculate_sale_price find the price of a $250 item with a 20% discount. "
    result2 = agent_executor.invoke({"input": task2})
    print("\nFinal Answer for Task 2:", result2['output'])
    print("-" * 50)