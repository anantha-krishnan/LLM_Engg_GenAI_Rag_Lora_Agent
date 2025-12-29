from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import operator

from crew_agent_tools import (
    find_matching_node_names,
    get_enriched_dossier,
    get_graph_schema,
    initialize_kg_toolbelt
)
from neo4j_kg_builder import Neo4jConnector
import global_vars

# Define the state that flows through the graph
class AnalystState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    chat_history: list[BaseMessage]
    user_question: str
    resolved_names: list[str]
    component_data: dict
    analysis_complete: bool
    iteration_count: int
    standalone_question: str

class LangGraphAnalystAgent:
    def __init__(self, llm_provider="openai"):
        if llm_provider == "openai":
            self.llm = ChatOpenAI(
                model_name=global_vars.model_openai_4o, 
                temperature=0.3,
                verbose=True
            )
        
        # Initialize Neo4j connector
        neo4j_connector = Neo4jConnector(
            global_vars.NEO4J_URI, 
            global_vars.NEO4J_USER, 
            global_vars.NEO4J_PASSWORD
        )
        
        # Initialize the global toolbelt with the connector
        initialize_kg_toolbelt(neo4j_connector=neo4j_connector, llm=self.llm)
        
        # Initialize tools - these are now LangChain-compatible
        self.tools = [
            find_matching_node_names,
            get_enriched_dossier,
            get_graph_schema
        ]
        
        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Create the graph
        self.graph = self._create_graph()
    def _history_aware_retrieval(self, state: AnalystState) -> dict:
        """ Reformulate question based on chat history. """
        print("\n---NODE (RAG): HISTORY-AWARE RETRIEVAL---")
        # ... (Your existing code for this function is perfect)
        question = state["user_question"]
        chat_history = state.get("chat_history", [])
        contextualize_q_system_prompt = """Given a chat history and the latest user question which might reference context
          in the chat history, formulate a standalone question which can be understood 
          without the chat history. Do NOT answer the question, just reformulate it if needed
          and otherwise return it as is. Carefully consider if the user is referring to any particular item in the chat history. Pick the names of those entities."""
        contextualize_q_prompt = ChatPromptTemplate.from_messages([("system", contextualize_q_system_prompt), MessagesPlaceholder(variable_name="chat_history"), ("human", "{question}")])
        history_retriever = contextualize_q_prompt | self.llm | StrOutputParser()
        retrieved_context = history_retriever.invoke({"chat_history": chat_history, "question": question})
        return {"standalone_question": retrieved_context or question}

    def _create_graph(self):
        """Create the LangGraph workflow"""
        workflow = StateGraph(AnalystState)
        workflow.add_node("history_aware_retrieval", self._history_aware_retrieval)
        # Add nodes
        workflow.add_node("manager", self._manager_node)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.add_node("synthesize", self._synthesize_node)
        
        # Set entry point
        workflow.set_entry_point("history_aware_retrieval")
        workflow.add_edge("history_aware_retrieval", "manager")
        # Add edges with conditional routing
        workflow.add_conditional_edges(
            "manager",
            self._should_continue,
            {
                "tools": "tools",
                "synthesize": "synthesize",
                "end": END
            }
        )
        
        # After tools run, go back to manager for decision
        workflow.add_edge("tools", "manager")
        
        # After synthesis, we're done
        workflow.add_edge("synthesize", END)
        
        return workflow.compile()
    
    def _manager_node(self, state: AnalystState):
        """Manager decides what to do next"""
        messages = state["messages"]
        iteration = state.get("iteration_count", 0)
        
        # Check iteration limit to prevent infinite loops
        if iteration > 10:
            return {
                "messages": [AIMessage(content="Maximum iterations reached. Synthesizing available data.")],
                "analysis_complete": True,
                "iteration_count": iteration + 1
            }
        
        # Manager prompt that encourages iterative analysis
        system_prompt = f"""You are an engineering research manager analyzing MotionSolve models.

Available tools:
1. find_matching_node_names - Find exact component names from fuzzy queries
2. get_enriched_dossier - Get detailed data for exact component names
3. get_graph_schema - Understand the graph structure

User's question: {state['user_question']}
Reformatted user's question based on history of chat: {state['standalone_question']}
Current iteration: {iteration + 1}
Data collected so far:
- Resolved names: {state.get('resolved_names', [])}
- Component data: {len(state.get('component_data', {}))} components analyzed

Your job: Decide what information you need next. You can:
- Search for component names if you don't have them yet
- Get detailed data for components
- Ask for related components to understand connections
- Request graph schema to understand relationships

Think step-by-step:
1. What do I know so far?
2. What information is still missing to answer the user's question?
3. What tool should I call next, or am I ready to synthesize?

If you have enough information to answer the question comprehensively, say "READY_TO_SYNTHESIZE" in your response.
Otherwise, call the appropriate tool to gather more information."""

        # Build messages list with system prompt
        messages_to_send = [
            HumanMessage(content=system_prompt)
        ] + messages
        
        # Get manager's decision
        response = self.llm_with_tools.invoke(messages_to_send)
        
        # Update state
        return {
            "messages": [response],
            "iteration_count": iteration + 1
        }
    
    def _should_continue(self, state: AnalystState):
        """Decide whether to continue gathering data or synthesize"""
        messages = state["messages"]
        last_message = messages[-1]
        
        # Check if analysis is complete
        if state.get("analysis_complete", False):
            return "synthesize"
        
        # Check if manager says ready to synthesize
        if "READY_TO_SYNTHESIZE" in last_message.content:
            return "synthesize"
        
        # If the last message has tool calls, execute them
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        
        # If no tool calls and not ready to synthesize, end
        return "end"
    
    def _synthesize_node(self, state: AnalystState):
        """Final synthesis of all gathered information"""
        messages = state["messages"]
        
        synthesis_prompt = f"""Based on all the information gathered through multiple iterations, 
provide a comprehensive answer to the user's question: {state['user_question']}

Data collected:
- Resolved component names: {state.get('resolved_names', [])}
- Detailed component data: {state.get('component_data', {})}

Provide a clear, well-structured engineering report that:
1. Directly answers the user's question
2. Explains the significance of the findings
3. Highlights any important relationships or patterns
4. Notes any limitations or areas needing more investigation

Format your response for an engineer."""

        response = self.llm.invoke([
            HumanMessage(content=synthesis_prompt)
        ])
        
        return {
            "messages": [response],
            "analysis_complete": True
        }
    
    def process_message(self, message: str, chat_history: list[BaseMessage]):
        """Process a user message through the graph"""
        
        # Initialize state
        initial_state = {
            "messages": [HumanMessage(content=message)],
            "chat_history": chat_history,
            "user_question": message,
            "resolved_names": [],
            "component_data": {},
            "analysis_complete": False,
            "iteration_count": 0,
            "standalone_question": ""
        }
        
        print("\n🚀 Starting iterative analysis workflow...\n")
        
        # Run the graph
        try:
            final_state = self.graph.invoke(initial_state)
            
            # Extract final answer
            final_message = final_state["messages"][-1]
            
            print(f"\n✅ Analysis complete after {final_state['iteration_count']} iterations\n")
            
            yield final_message.content
            
        except Exception as e:
            yield f"Error during analysis: {str(e)}"
    
    def stream_message(self, message: str, chat_history: list[BaseMessage]):
        """Stream the analysis process"""
        initial_state = {
            "messages": [HumanMessage(content=message)],
            "user_question": message,
            "resolved_names": [],
            "component_data": {},
            "analysis_complete": False,
            "iteration_count": 0
        }
        
        # Stream events from the graph
        for event in self.graph.stream(initial_state, stream_mode="values"):
            if "messages" in event and event["messages"]:
                last_message = event["messages"][-1]
                
                if isinstance(last_message, AIMessage):
                    yield f"\n{'='*60}\n"
                    yield f"Iteration {event.get('iteration_count', 0)}: "
                    
                    # Show content if available
                    if hasattr(last_message, 'content') and last_message.content:
                        yield f"{last_message.content}\n"
                    
                    # Show tool calls if available
                    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                        tool_names = [tc.get('name', 'unknown') for tc in last_message.tool_calls]
                        yield f"→ Calling tools: {tool_names}\n"
                
                elif isinstance(last_message, ToolMessage):
                    yield f"✓ Tool result received\n"