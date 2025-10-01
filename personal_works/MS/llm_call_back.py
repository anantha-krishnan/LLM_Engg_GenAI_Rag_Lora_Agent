# file: llm_call_back.py


import re
from typing import List, Any, Dict

from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.outputs import LLMResult

class VerboseToUIHandler(BaseCallbackHandler):
    """
    Callback handler that captures the agent's verbose output for the UI.
    It formats the thoughts, actions, and observations for clarity.
    """
    def __init__(self):
        self.logs = []
        self.ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        """Log the start of a chain."""
        # We only want to show the top-level agent start, not inner chains like LLMChain.
        if serialized and serialized.get("name") == "AgentExecutor":
             self.logs.append("\n> Entering new AgentExecutor chain...")

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Formats and logs the agent's thought and action."""
        # action.log is the raw thought process from the LLM
        thought = self.ansi_escape.sub('', action.log).strip()
        
        # Extract the tool and input for cleaner logging
        tool = action.tool
        tool_input = action.tool_input
        
        # Format the output
        log_entry = (
            f"{thought}\n"
            f"Action: `{tool}`\n"
            f"Action Input: `{tool_input}`"
        )
        self.logs.append(log_entry)

    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        """Formats and logs the tool's observation."""
        clean_output = self.ansi_escape.sub('', str(output))
        observation = f"Observation: {clean_output}"
        self.logs.append(observation)

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Formats and logs the agent's final answer."""
        thought_and_answer = self.ansi_escape.sub('', finish.log).strip()
        self.logs.append(thought_and_answer)
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Log the end of the chain."""
        if "AgentExecutor" in str(kwargs.get("run_id", "")): # A simple check for the top-level agent chain
            self.logs.append("\n> Finished chain.")

    def add_log_entry(self, text: str):
        """Allows for manually adding log entries from outside the callback system."""
        self.logs.append(text)

    def get_logs(self) -> str:
        """Return all captured logs as a single formatted string."""
        return "\n\n".join(self.logs) # Use double newline for better readability