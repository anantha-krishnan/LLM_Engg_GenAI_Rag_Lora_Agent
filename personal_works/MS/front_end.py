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
from code_execution_agent import create_code_executor, validate_user_function, register_user_function, save_helper_function
import global_vars
from session_state import OnboardingState, SESSION_STATE, reset_session
from tool_input_parser import RobustTool
from llm_call_back import StreamingVerboseToUIHandler 
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
        self.current_callbacks = None

    def _parse_specialist_output(self, agent_output: str) -> str:
        match = re.search(r"Execution Result:\s*(\{.*\})", str(agent_output), re.DOTALL)
        return match.group(1) if match else agent_output

    def _tool_logic(self,func_name: str,**kwargs):
        """
        This wrapper calls the specialist agent. It receives clean kwargs
        because the RobustStructuredTool has already parsed and validated them.
        """
        # if "session_work_dir" not in kwargs:
            # kwargs["session_work_dir"] = SESSION_STATE["work_dir"]
        prompt_for_specialist = f"Run the function '{func_name}' with these arguments in a JSON string: {json.dumps(kwargs)}"
        log_callback = self.current_callbacks[0] if self.current_callbacks else None

        if log_callback:
        # Manually add a log entry to clearly mark the start of the inner agent's work
            log_callback.add_log_entry(f"\n> [Orchestrator] Delegating to Specialist Agent to run tool: `{func_name}`...")

        specialist_result = self.code_agent_executor.invoke({"input": prompt_for_specialist},
                                                            config={"callbacks": self.current_callbacks})
        
        if log_callback:
        # Manually add a log entry to mark the end of the inner agent's work
            log_callback.add_log_entry(f"> [Orchestrator] Specialist Agent finished.")
        return self._parse_specialist_output(specialist_result['output'])

    
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
        
        
        tool_func = lambda **kwargs: self._tool_logic(func_name=func_name, **kwargs)
        new_tool = RobustTool(name=func_name, description=docstring, func=tool_func, args_schema=args_schema)

        self.orchestrator_tools.append(new_tool)
        return f"✅ Tool '{func_name}' registered successfully."

    def create_orchestrator(self, instructions: str):
        if not self.orchestrator_tools:
            tool_instructions = "No tools have been registered."
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


    async def run_orchestrator(self, goal: str, uploaded_file_paths: List[str] = None):
        if not self.orchestrator_agent_executor:
            yield "❌ Orchestrator has not been created yet.", "No logs available."
            return
            
        if uploaded_file_paths:
            filenames = [os.path.basename(p) for p in uploaded_file_paths]
            work_dir = SESSION_STATE['work_dir']
            # The prompt now clearly lists all available files.
            goal += (f" The user has uploaded the following files: {filenames}. "
                     f"These files are all located in the session's working directory at '{work_dir}'. "
                     f"The user's prompt will specify which file is the primary input. "
                     "You must pass the name of the primary input file to the appropriate tool argument.")
            
        print(f"\n--- EXECUTING ORCHESTRATOR with GOAL: {goal} ---")
        log_queue = asyncio.Queue()
        log_callback = StreamingVerboseToUIHandler(log_queue)
        self.current_callbacks = [log_callback]

        final_output = "Agent run did not produce a final output."
        full_log_history = []

        # Run the agent in a background task so we can listen to the queue simultaneously
        async def agent_task():
            nonlocal final_output
            async for chunk in self.orchestrator_agent_executor.astream(
                                                                    {"input": goal},
                                                                    config={"callbacks": self.current_callbacks}
                                                                ):
                if 'output' in chunk:
                    final_output = chunk['output']
            
            # Signal that the agent is done by putting a sentinel value in the queue
            await log_queue.put(None)

        task = asyncio.create_task(agent_task())

        while True:
            log_item = await log_queue.get()
            if log_item is None:
                # Sentinel value received, agent is finished
                break
            
            full_log_history.append(log_item)
            # Yield the accumulated log history for the UI
            yield final_output, f"```log\n{''.join(full_log_history)}\n```"

        await task # Ensure the agent task is complete
        self.current_callbacks = None
        
        # Yield the final, complete state
        yield final_output, f"```log\n{''.join(full_log_history)}\n```"


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

async def chat_responder(message: str, history: list, uploaded_files: List[str]):
    state = SESSION_STATE["state"]
    response = None
    logs = "No log available."
    if state == OnboardingState.AGENT_READY:
        # The file is processed inside user_interaction before this is called
        # Instead of getting a final response, we iterate over the stream
        async for response, logs in agent_system.run_orchestrator(message, uploaded_files):
            logs = f"```log\n{logs}\n```"
            yield response, logs

    if state == OnboardingState.START:
        SESSION_STATE["state"] = OnboardingState.AWAITING_INSTRUCTIONS
        response = "Hello! Let's build a specialized agent. First, please provide its core instructions (the step-by-step logic it should follow). When you're finished, type `/done` on a new line."
    if state in [OnboardingState.AWAITING_INSTRUCTIONS, OnboardingState.COLLECTING_INSTRUCTIONS]:
        if message.strip().lower() == "/done":
            SESSION_STATE["state"] = OnboardingState.AWAITING_HELPERS
            response = "Instructions saved. Now, let's add the helper Python functions that will be needed for your functions.\n\n"
        else:
            SESSION_STATE["instructions"] += message + "\n"
            SESSION_STATE["state"] = OnboardingState.COLLECTING_INSTRUCTIONS
            response = "Instruction line added. Continue adding to the instructions or type `/done` to finish."
    if state in [OnboardingState.AWAITING_HELPERS, OnboardingState.COLLECTING_HELPERS]:
        if message.strip().lower() == "/done":
            SESSION_STATE["state"] = OnboardingState.AWAITING_FUNCTIONS
            msg="Now, please provide the main Python functions that the orchestrator will use as tools.\n\n"
            msg+="Remember the rules:\n"
            msg+="1. All arguments must have type hints (e.g., `name: str`).\n"
            msg+="2. The function must have a return type hint of `-> dict:` or `-> Dict:`.\n"
            msg+="3. The function must actually return a dictionary.\n\n"
            msg+="Please provide the code for the first function."
            response = msg
        else:
            # Validate helper functions similarly if desired
            is_valid, message, tree = validate_user_function(message)
            if not is_valid:
                print(f"   [Validation] Failed: {message}")
            else:                
                print(f"   [Validation] {message}")
            # save helper function regardless of validation for now as it will be used internally by user directly
            save_helper_function(tree)
            SESSION_STATE["state"] = OnboardingState.COLLECTING_HELPERS
            response = "Helper function added. Continue adding helper functions or type `/done` to finish."
    if state in [OnboardingState.AWAITING_FUNCTIONS, OnboardingState.COLLECTING_FUNCTIONS]:
        if message.strip().lower() == "/done":
            if not agent_system.orchestrator_tools:
                response = "❌ You haven't registered any functions. Please add at least one function before finishing."
            creation_response = agent_system.create_orchestrator(SESSION_STATE["instructions"])
            if "✅" in creation_response:
                SESSION_STATE["state"] = OnboardingState.AGENT_READY
                response = (f"{creation_response}\n\n"
                f"**The specialized agent is now ready. What is its first high-level goal?**")
            else:
                response = creation_response
        else:
            tool_response = agent_system.register_tool_for_orchestrator(message)
            SESSION_STATE["state"] = OnboardingState.COLLECTING_FUNCTIONS
            response = f"{tool_response}\n\nPlease provide the next function, or type `/done` when you have added all necessary functions."
    if not response:
        response = "An unknown error occurred with the conversation state."
    yield response, logs

with gr.Blocks(theme=gr.themes.Soft(), title="Dynamic Agent Orchestrator") as demo:
    gr.Markdown("# Dynamic Agent Orchestrator\n A conversational interface to build and command specialized AI agents.")
    with gr.Row():
        chatbot = gr.Chatbot(label="Conversation", height=600)
    with gr.Row():
        with gr.Accordion("Agent's Thought Process", open=False):
            agent_log = gr.Markdown("Agent logs will appear here...")
    with gr.Row():
        msg_textbox = gr.Textbox(label="Your Message", placeholder="Start by describing the agent you want to build...", scale=7)
        file_uploader = gr.File(label="Upload Input Files (XML, etc.)",
            file_count="multiple",  # This is the key change
            scale=1)
    clear_button = gr.Button("Clear Conversation & Start New Session")

    async def user_interaction(user_message, chat_history, uploaded_file):
        # Process file first, get its permanent path in the session directory
        file_path_on_server = process_uploaded_file(uploaded_file)
        chat_history.append((user_message, ""))
        bot_message = ""
        log_output = ""
        final_bot_message = None
        # Get the response from our stateful chat function, passing the path
        async for partial_bot_message, partial_logs in chat_responder(user_message, chat_history, file_path_on_server):
            bot_message = partial_bot_message
            verbose_logs = partial_logs
            chat_history[-1] = (user_message, bot_message)
            yield "", chat_history, verbose_logs, None

    msg_textbox.submit(user_interaction,
                       [msg_textbox, chatbot, file_uploader],
                       [msg_textbox, chatbot, agent_log, file_uploader])
    # The clear button now also creates a new working directory
    clear_button.click(lambda: (reset_session(), None, None),
                       None,
                       [chatbot,file_uploader,agent_log], queue=False)

app = fastapi.FastAPI(title="Agent Orchestrator API")
app = gr.mount_gradio_app(app, demo, path="/")
    
if __name__ == "__main__":
    print("="*50)
    print("🚀 Your agent application is ready!")
    print("1. Run the websocket bridge in another terminal: python 2_websocket_bridge.py")
    print("2. Run this app with uvicorn: uvicorn front_end:app --reload --host 0.0.0.0 --port 8000")
    print("3. Open your browser to http://127.0.0.1:8000")
    print("="*50)

"""
Follow this exact algorithm:
You are an expert optimization engineer for Altair MotionSolve. Your goal is to find the optimal (largest) solver time step (h_max) for a user's model that keeps the simulation result difference below a specified threshold.
1. **Golden Run**: First, perform a baseline simulation with a very small h_max (e.g., 1e-5) and mode='PRE' to get a "golden" result file. This is your ground truth. Call the tool 'analyze_simulation_results' with mode='PRE' to establish the baseline
2. **Iterate and Compare**: Run a new simulation with test_h_max and mode='NORM'. In each test Analyze the dictionary returned by the tool. If the value for the key 'percentage_difference' is less than 5.0, your job is done. Your final answer is the last h_max you used.
3. **Decide and Adjust**:
 If the difference is less than the user's threshold (e.g., 5%): This test_h_max is valid. It means you might be able to use an even larger step. Therefore, store this as a potential answer and set the lower bound of your search range to test_h_max.
 If the difference is greater than or equal to the threshold: This test_h_max is too large and invalid. You must use a smaller step. Set the upper bound of your search range to test_h_max.
4. **Termination**: Continue this process for a fixed number of iterations (e.g., 10-15 is usually enough for good precision) or until the search range is very small.
5. **Report**: Once finished, clearly state the largest h_max you found that satisfied the condition and conclude your work.
"""  
# you are specialist at addition. you will receive two numbers from user. Send it to the tool for addition. Reply with the result from the tool
"""
The file c11x001m.xml is the input to motionsolve. I have already hard coded the qa working directory in the tool 'analyze_simulation_results'. This directory contains the necessary folder structure. The file c11x001m.xml is present at the correct location as required by the tool. Your goal is to find the optimal (largest) solver time step (`h_max`) for this input file that keeps the simulation result difference below 5%. First run with mode='PRE', h_max=0.001, xml_filename='c11x001m.xml' to get the golden reference result, then iteratively run with mode='NORM' and different `h_max` values to find the largest acceptable `h_max`. 
"""
