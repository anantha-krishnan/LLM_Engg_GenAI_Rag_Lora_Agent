from pathlib import Path
from typing import List, TypedDict, Generator, Optional, Any

from neo4j_kg_builder import Neo4jConnector
from agent_tools_2 import ToolBelt
import global_vars
from global_vars import GraphState

from langchain_core.messages import BaseMessage
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser


from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver




from typing import List, Generator, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import json

# Import your existing class
from action_step_executor_analyst import ActionStepExecutorAnalyst
import global_vars

class SupervisorAgent:
    def __init__(self, connector: Neo4jConnector):
        # 1. Initialize the Brain (for routing/reformulation)
        self.llm = ChatOpenAI(model=global_vars.model_openai_4omini, temperature=0)
        
        # 2. Initialize the Memory (The Context Artifact)
        self.last_context = {
            "key_facts": [],
            "entities": [],
            "hypothesis": ""
        }
        
        # 3. Initialize the Sub-Agent
        # We keep it persistent so it holds its own Neo4j connection
        self.analyst = ActionStepExecutorAnalyst(connector)

    def _decide_action(self, user_input: str, history: List) -> Dict:
        """
        Internal helper: Decides if we chat or call the analyst.
        Returns JSON: {"action": "chat" | "analyze", "payload": "..."}
        """
        formatted_history = ""
        if history:
            # Assuming history is a list of LangChain BaseMessage objects or dicts
            # Adjust logic depending on your history format
            recent_msgs = history[-6:] 
            for msg in recent_msgs:
                role = "User" if msg.type == "human" else "AI"
                formatted_history += f"{role}: {msg.content}\n"
        system_prompt = f"""
        You are the Supervisor MBD analyst. The user has asked a question `USER INPUT`. You need to respond to it. You have two options:
        DECISION LOGIC:
        1. CHAT with the user directly using `chat_history`; If you have sufficient context to answer their query.
        2. DELEGATE to the MBD Analyst agent for technical analysis; If the user query requires new information or deep analysis. In this case you must REFORMULATE the query to be STANDALONE, specific and clear, avoiding pronouns.
        Add the necessary context from `chat_history`.
        
        CHAT_HISTORY: {formatted_history}
        
        USER INPUT: "{user_input}"
        
        TASK STEPS:
        1. Analyze the USER INPUT in the context of CHAT_HISTORY.
        2. Decide whether to CHAT or DELEGATE to the Analyst.
        3. If DELEGATE, REFORMULATE the query to be clear and standalone.
        4. RESPOND in strict JSON format.
        
        OUTPUT JSON ONLY: {{ "action": "...", "payload": "..." }}
        """
        
        response = self.llm.invoke(system_prompt).content
        try:
            # Simple cleaning in case LLM adds markdown
            clean_json = response.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_json)
        except:
            # Fallback
            return {"action": "chat", "payload": "I didn't understand. Could you rephrase?"}

    def process_message(self, message: str, chat_history: List) -> Generator[dict, None, None]:
        """
        The Main Generator called by your UI.
        It yields events just like the sub-agent did.
        """
        
        # Step 1: Supervisor Decision (Fast, non-streaming)
        decision = self._decide_action(message, chat_history)
        
        if decision["action"].lower() == "chat":
            # Direct reply, just yield text
            yield {"type": "text", "data": decision["payload"]}
            return

        elif decision["action"].lower() == "delegate":
            # We are calling the sub-agent!
            reformulated_query = decision["payload"]
            
            # Yield a status update so user knows we are switching agents
            yield {"type": "status", "data": f"Supervisor: Delegating to Analyst with query: '{reformulated_query}'..."}
            
            # Step 2: Call the Sub-Agent and STREAM its results
            # This is the "Proxy" part. We loop through the sub-agent's generator.
            sub_agent_generator = self.analyst.process_message(reformulated_query, [])
            
            final_artifact_captured = None

            for event in sub_agent_generator:
                # --- A. HANDLE PASS-THROUGH EVENTS ---
                # Events like 'plot', 'status', 'plan', 'chain' go DIRECTLY to UI
                if event["type"] in ["status", "plot", "plan", "chain"]:
                    yield event
                
                # --- B. HANDLE TEXT CHUNKS ---
                elif event["type"] == "text":
                    yield event

                # --- C. CAPTURE CONTEXT (Do not yield to UI) ---
                # This assumes you updated your Analyst to yield this special type
                elif event["type"] == "final_answer":
                    final_data = event["data"]["final_answer"]
                    final_answer = final_data.get("final_answer", "")
                    # get "context_artifact" if present
                    final_artifact_captured = final_data.get("context_artifact", None)
                    # get "concise_summary" and "key_facts" from final_artifact_captured
                    concise_summary = final_artifact_captured.get("concise_summary", "")
                    key_facts = final_artifact_captured.get("key_facts", [])
                    # Yield the final answer to UI
                    yield {"type": "text", "data": final_answer}
                
                # --- D. FALLBACK ---
                else:
                    yield event

            # Step 3: Update Supervisor Memory
            if final_artifact_captured:
                self.last_context = final_artifact_captured
                print(f"Supervisor Memory Updated: {len(self.last_context['key_facts'])} facts stored.")
            else:
                # Fallback if sub-agent didn't yield artifacts (e.g. error)
                pass
if __name__ == "__main__":
    # Example usage
    neo4j_connector = Neo4jConnector(
    uri=global_vars.NEO4J_URI,
    user=global_vars.NEO4J_USER,
    password=global_vars.NEO4J_PASSWORD
    )
    qa_analyst_agent = SupervisorAgent(neo4j_connector)
    qa_analyst_agent.save_graph(Path(__file__).parent / "qa_analyst_agent_h_graph.png")        