# In file: llm_call_back.py (or wherever VerboseToUIHandler is defined)

import re
import asyncio
from typing import Dict, Any
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish

class StreamingVerboseToUIHandler(BaseCallbackHandler):
    """
    Callback handler that puts agent's verbose output into an asyncio.Queue.
    """
    def __init__(self, queue: asyncio.Queue):
        super().__init__()
        self.queue = queue
        self.ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        self.log_buffer = []

    def _push_to_queue(self):
        """Pushes the buffered log content to the queue as a single update."""
        if self.log_buffer:
            # We join with double newlines for better spacing in the UI
            log_text = "\n\n".join(self.log_buffer)
            self.queue.put_nowait(log_text)
            self.log_buffer = []

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        thought = self.ansi_escape.sub('', action.log).strip()
        log_entry = (
            f"{thought}\n"
            f"Action: `{action.tool}`\n"
            f"Action Input: `{action.tool_input}`\n"
        )
        self.log_buffer.append(log_entry)
        self._push_to_queue()

    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        clean_output = self.ansi_escape.sub('', str(output))
        observation = f"Observation: {clean_output}\n"
        self.log_buffer.append(observation)
        self._push_to_queue()

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        thought_and_answer = self.ansi_escape.sub('', finish.log).strip()
        self.log_buffer.append(thought_and_answer)
        self._push_to_queue()
    
    def add_log_entry(self, text: str):
        """Allows for manually adding log entries from outside the callback system."""
        self.log_buffer.append(text)
        self._push_to_queue()