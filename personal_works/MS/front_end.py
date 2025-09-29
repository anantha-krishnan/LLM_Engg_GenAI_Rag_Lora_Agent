# File: 5_main_app.py

import gradio as gr
import fastapi
import asyncio
from typing import List, Any, Tuple, Dict
import os
import shutil
import ast
import re
import json
from textwrap import dedent

from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain import hub
from langchain_openai import ChatOpenAI  
# from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import create_model


# Local imports
from code_execution_agent import create_code_executor, validate_user_function, register_user_function
import global_vars
from session_state import OnboardingState, SESSION_STATE, reset_session
from tool_input_parser import RobustTool
# Initialize the first session on startup
reset_session()


class HierarchicalAgentSystem:
    def __init__(self):
        print("Initializing Hierarchical Agent System...")
        self.code_agent_executor = create_code_executor()
        print("✅ Specialist Code Execution Agent is ready.")
        self.orchestrator_tools: List[Tool] = []
        self.orchestrator_agent_executor = None
        self.is_orchestrator_created = False

    def _parse_specialist_output(self, agent_output: str) -> str:
        match = re.search(r"Execution Result:\s*(\{.*\})", str(agent_output), re.DOTALL)
        return match.group(1) if match else agent_output


    def register_tool_for_orchestrator(self, function_code: str):
        print(f"\n--- Attempting to register new Orchestrator tool ---")
        is_valid, message, tree = validate_user_function(function_code)
        if not is_valid:
            print(f"   [Validation] Failed: {message}")
            return f"❌ Validation Failed: {message}"
        print(f"   [Validation] {message}")
        try:
            func_def = tree.body[-1]
            func_name = func_def.name
            docstring = ast.get_docstring(func_def) or f"Runs the '{func_name}' function."
            pydantic_fields = {arg.arg: (eval(ast.unparse(arg.annotation)), ...) for arg in func_def.args.args}
            args_schema = create_model(f"{func_name}Input", **pydantic_fields)
        except Exception as e:
            return f"❌ Error creating tool schema: {e}"
        
        result = register_user_function(tree)
        if "Error:" in result:
            return f"❌ Specialist failed to register function: {result}"
        print(f"   [Specialist] Successfully registered '{func_name}'.")
        
        def _tool_logic(**kwargs):
            """
            This wrapper calls the specialist agent. It receives clean kwargs
            because the RobustStructuredTool has already parsed and validated them.
            """
            if "session_work_dir" not in kwargs:
                kwargs["session_work_dir"] = SESSION_STATE["work_dir"]
            
            prompt_for_specialist = f"Run the function '{func_name}' with these arguments in a JSON string: {json.dumps(kwargs)}"
            specialist_result = self.code_agent_executor.invoke({"input": prompt_for_specialist})
            return self._parse_specialist_output(specialist_result['output'])

        new_tool = RobustTool(name=func_name, description=docstring, func=_tool_logic, args_schema=args_schema)
        
        self.orchestrator_tools.append(new_tool)
        return f"✅ Tool '{func_name}' registered successfully."

    def create_orchestrator(self, instructions: str):
        if not self.orchestrator_tools:
            return "❌ Cannot create orchestrator: No tools have been registered."
        print(f"\n--- Creating new Orchestrator Agent ---")
        tool_instructions = (
            "When you use a tool, you MUST provide the arguments as a flat JSON dictionary. DO NOT add comments or extra text."
            "Do NOT nest the entire dictionary of arguments inside the value of the first argument."
            "\n\nCORRECT FORMAT:\nAction Input: {\"arg1\": \"value1\", \"arg2\": 123}"
            "\n\nINCORRECT FORMAT:\nAction Input: {\"arg1\": \"{\"arg1\": \"value1\", \"arg2\": 123}\"}"
        )
        
        # Combine the user's instructions with our formatting rule
        full_instructions = f"{instructions}\n\n--- TOOL USAGE RULES ---\n{tool_instructions}"
        
        react_prompt = hub.pull("hwchase17/react")
        prompt = react_prompt.partial(instructions=dedent(full_instructions))
        llm = ChatOpenAI(temperature=0, model=global_vars.model_openai_4o)
        agent = create_react_agent(llm, self.orchestrator_tools, prompt)
        self.orchestrator_agent_executor = AgentExecutor(
            agent=agent, tools=self.orchestrator_tools, verbose=True,
            max_iterations=20, handle_parsing_errors=True
        )
        self.is_orchestrator_created = True
        return "✅ Orchestrator Agent created successfully. You can now give it a goal."


    def run_orchestrator(self, goal: str, uploaded_file_paths: List[str] = None):
        if not self.orchestrator_agent_executor:
            return "❌ Orchestrator has not been created yet."
            
        if uploaded_file_paths:
            filenames = [os.path.basename(p) for p in uploaded_file_paths]
            work_dir = SESSION_STATE['work_dir']
            # The prompt now clearly lists all available files.
            goal += (f" The user has uploaded the following files: {filenames}. "
                     f"These files are all located in the session's working directory at '{work_dir}'. "
                     f"The user's prompt will specify which file is the primary input. "
                     "You must pass the full path of the primary input file to the appropriate tool argument.")
            
        print(f"\n--- EXECUTING ORCHESTRATOR with GOAL: {goal} ---")
        result = self.orchestrator_agent_executor.invoke({"input": goal})
        return result.get('output', "Completed with no output.")


# --- System Singleton & UI Logic (with updated file handling) ---
agent_system = HierarchicalAgentSystem()

def process_uploaded_file(file_objs: List[Any]) -> List[str]:
    """Saves the uploaded file to the CURRENT session's working directory."""
    if not file_objs: return []
    saved_file_paths = []
    # Get the current session's working directory
    work_dir = SESSION_STATE.get("work_dir")
    if not work_dir:
        # Fallback in case session wasn't initialized, though it should be.
        reset_session()
        work_dir = SESSION_STATE["work_dir"]

    # Gradio gives a temp file path; we copy it to our session directory.
    for file_obj in file_objs:
        file_path = os.path.join(work_dir, os.path.basename(file_obj.name))
        shutil.copyfile(file_obj.name, file_path)

        print(f"File uploaded and saved to session directory: {file_path}")
        saved_file_paths.append(file_path)
    return saved_file_paths

def chat_responder(message: str, history: list, uploaded_files: List[str]) -> Tuple[str, Any]:
    state = SESSION_STATE["state"]
    
    if state == OnboardingState.AGENT_READY:
        # The file is processed inside user_interaction before this is called
        response = agent_system.run_orchestrator(message, uploaded_files) # uploaded_files is the path now
        return response, None

    if state == OnboardingState.START:
        SESSION_STATE["state"] = OnboardingState.AWAITING_INSTRUCTIONS
        return "Hello! Let's build a specialized agent. First, please provide its core instructions (the step-by-step logic it should follow). When you're finished, type `/done` on a new line.", None
    if state in [OnboardingState.AWAITING_INSTRUCTIONS, OnboardingState.COLLECTING_INSTRUCTIONS]:
        if message.strip().lower() == "/done":
            SESSION_STATE["state"] = OnboardingState.AWAITING_FUNCTIONS
            return ("Instructions saved. Now, let's add the Python functions this agent will use as tools.\n\n"
                    "**IMPORTANT RULES for functions:**\n"
                    "1. All arguments must have type hints (e.g., `name: str`).\n"
                    "2. The function must have a return type hint of `-> dict:` or `-> Dict:`.\n"
                    "3. The function must actually return a dictionary.\n\n"
                    "Please provide the code for the first function."), None
        else:
            SESSION_STATE["instructions"] += message + "\n"
            SESSION_STATE["state"] = OnboardingState.COLLECTING_INSTRUCTIONS
            return "Instruction line added. Continue adding to the instructions or type `/done` to finish.", None
    if state in [OnboardingState.AWAITING_FUNCTIONS, OnboardingState.COLLECTING_FUNCTIONS]:
        if message.strip().lower() == "/done":
            if not agent_system.orchestrator_tools:
                return "❌ You haven't registered any functions. Please add at least one function before finishing.", None
            creation_response = agent_system.create_orchestrator(SESSION_STATE["instructions"])
            if "✅" in creation_response:
                SESSION_STATE["state"] = OnboardingState.AGENT_READY
                return (f"{creation_response}\n\n"
                        "**The specialized agent is now ready. What is its first high-level goal?**"), None
            else:
                return creation_response, None
        else:
            tool_response = agent_system.register_tool_for_orchestrator(message)
            SESSION_STATE["state"] = OnboardingState.COLLECTING_FUNCTIONS
            return f"{tool_response}\n\nPlease provide the next function, or type `/done` when you have added all necessary functions.", None
    return "An unknown error occurred with the conversation state.", None

with gr.Blocks(theme=gr.themes.Soft(), title="Dynamic Agent Orchestrator") as demo:
    gr.Markdown("# Dynamic Agent Orchestrator\n A conversational interface to build and command specialized AI agents.")
    chatbot = gr.Chatbot(label="Conversation", height=600)
    with gr.Row():
        msg_textbox = gr.Textbox(label="Your Message", placeholder="Start by describing the agent you want to build...", scale=7)
        file_uploader = gr.File(label="Upload Input Files (XML, etc.)",
            file_count="multiple",  # This is the key change
            scale=1)
    clear_button = gr.Button("Clear Conversation & Start New Session")

    def user_interaction(user_message, chat_history, uploaded_file):
        # Process file first, get its permanent path in the session directory
        file_path_on_server = process_uploaded_file(uploaded_file)
        
        # Get the response from our stateful chat function, passing the path
        return_st = chat_responder(user_message, chat_history, file_path_on_server)
        bot_message = return_st[0]

        chat_history.append((user_message, bot_message))
        return "", chat_history, None # Clear inputs

    msg_textbox.submit(user_interaction, [msg_textbox, chatbot, file_uploader], [msg_textbox, chatbot, file_uploader])
    # The clear button now also creates a new working directory
    clear_button.click(lambda: (reset_session(), None), None, [chatbot,file_uploader], queue=False)

app = fastapi.FastAPI(title="Agent Orchestrator API")
app = gr.mount_gradio_app(app, demo, path="/")
    
if __name__ == "__main__":
    print("="*50)
    print("🚀 Your agent application is ready!")
    print("1. Run the websocket bridge in another terminal: python 2_websocket_bridge.py")
    print("2. Run this app with uvicorn: uvicorn 5_main_app:app --reload")
    print("3. Open your browser to http://127.0.0.1:8000")
    print("="*50)

"""
Follow this exact algorithm:
You are an expert optimization engineer for Altair MotionSolve. Your goal is to find the optimal (largest) solver time step (h_max) for a user's model that keeps the simulation result difference below a specified threshold.
1. **Golden Run**: First, perform a baseline simulation with a very small h_max (e.g., 1e-5) to get a "golden" result file. This is your ground truth. Call the tool 'analyze_simulation_results' with mode='PRE' to establish the baseline
2. **Iterate and Compare**: Run a new simulation with test_h_max and mode='NORM'. In each test Analyze the dictionary returned by the tool. If the value for the key 'percentage_difference' is less than 5.0, your job is done. Your final answer is the last h_max you used.
3. **Decide and Adjust**:
 If the difference is less than the user's threshold (e.g., 5%): This test_h_max is valid. It means you might be able to use an even larger step. Therefore, store this as a potential answer and set the lower bound of your search range to test_h_max.
 If the difference is greater than or equal to the threshold: This test_h_max is too large and invalid. You must use a smaller step. Set the upper bound of your search range to test_h_max.
4. **Termination**: Continue this process for a fixed number of iterations (e.g., 10-15 is usually enough for good precision) or until the search range is very small.
5. **Report**: Once finished, clearly state the largest h_max you found that satisfied the condition and conclude your work.
"""  

"""
The file FullVehNoDriver_saved.xml is the input to motionsolve. Your goal is to find the optimal (largest) solver time step (`h_max`) for this input file that keeps the simulation result difference below 5%.
"""
