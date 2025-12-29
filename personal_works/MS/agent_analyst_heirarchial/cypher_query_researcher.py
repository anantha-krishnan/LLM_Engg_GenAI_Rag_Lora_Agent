from pathlib import Path
from typing import List, TypedDict, Generator, Optional, Any
from neo4j_kg_builder import Neo4jConnector
from agent_tools_2 import ToolBelt
import global_vars
import operator

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
from langchain_core.documents import Document
from pydantic.v1 import BaseModel, Field
from langchain_community.document_loaders import WebBaseLoader
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts.chat import ChatPromptTemplate
from langgraph.prebuilt import ToolNode

from typing import List, Tuple, Annotated, TypedDict, Union, Dict

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
# Assuming these are imported from your existing files
# from neo4j_kg_builder import Neo4jConnector
# from agent_tools_2 import ToolBelt
import global_vars

# --- 1. STATE DEFINITION ---
class ExecutionStateCypher(TypedDict):
    query: str                       # The original research query
    resolved_entities: Annotated[List[str], operator.add]       # Actual names found via fuzzy search
    cypher_query: str                # The generated Cypher
    results: str                     # Results from Neo4j
    errors: List[str]                # Any Cypher execution errors
    iteration_count: int
    schema_context: str

class cypher_query_researcher:
    def __init__(self):
        llm = ChatOpenAI(
                model_name=global_vars.model_openai_4omini,
                #global_vars.model_openai_4omini,
                #openai_api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
                #openai_api_key=global_vars.google_api_key,
                temperature=0.3,
                streaming=True,
            )
        neo4j_connector = Neo4jConnector(
                uri=global_vars.NEO4J_URI,
                user=global_vars.NEO4J_USER,
                password=global_vars.NEO4J_PASSWORD
            )
        tool_belt = ToolBelt(neo4j_connector) 
        self.neo4j_connector = neo4j_connector
        self.tool_belt = tool_belt
        self.llm = llm
        
        # Initialize ToolNode and the Graph
        self.tools = self._get_tools()
        self.tool_node = ToolNode(self.tools)
        self.subgraph = self._create_sub_graph()

    def _get_tools(self):
        @tool
        def fuzzy_search_node(query: str):
            """Search for real node names in the KG when the user's name is approximate."""
            return self.tool_belt.find_matching_node_names(query)
        
        @tool
        def find_nodes_by_type_Body(node_type: str="Body"):
            """Find all the "Body" type nodes in the KG."""
            return self.tool_belt.find_nodes_by_type(node_type)
        @tool
        def find_nodes_by_type_PostRequest(node_type: str="PostRequest"):
            """Find all the "PostRequest" type nodes in the KG."""
            return self.tool_belt.find_nodes_by_type(node_type)
        @tool
        def find_nodes_by_type_OutputComponent(node_type: str="OutputComponent"):
            """Find all the "OutputComponent" type nodes in the KG."""
            return self.tool_belt.find_nodes_by_type(node_type)
        @tool
        def get_node_details(entity_name: str):
            """Get properties and documentation for a specific node name."""
            return self.tool_belt.get_enriched_dossier(entity_name)
        @tool
        def get_all_output_relationships():
            """Get all nodes connected through forming a MEASURES_OUTPUT relationship in the KG.
            This means which PostRequest nodes measure which nodes."""
            # execute the query \n
            cypher_query = """
            MATCH p = (n)-[r:MEASURES_OUTPUT]-(m)
            RETURN p
            """
            records = self.neo4j_connector.query(cypher_query)
            return self.neo4j_connector.format_results_to_text(records)
        @tool
        def get_all_joint_relationships():
            """Get all nodes connected through a CONNECTS_TO relationship in the KG.
            This means all the Bodies connected through joints."""            
            cypher_query = """
            MATCH p = (n)-[r:CONNECTS_TO]-(m)
            RETURN p
            """
            records = self.neo4j_connector.query(cypher_query)
            return self.neo4j_connector.format_results_to_text(records)
        return [fuzzy_search_node, get_node_details, find_nodes_by_type_OutputComponent, find_nodes_by_type_Body,
                get_all_output_relationships, get_all_joint_relationships]

    # --- NODES ---

    def _entity_resolver_node(self, state: ExecutionStateCypher):
        """Step 1: Map user terms to real Graph names."""
        print("--- RESEARCHER: RESOLVING ENTITIES ---")
        schema = self.neo4j_connector.get_full_graph()
        prompt = ChatPromptTemplate.from_template(
            "You are an expert MBD Knowledge Graph researcher. You need to identify how to attack the user's research query. You can choose from the following tools.\n"
            "The user message: {query}\n"
            "**Tools Available:**\n"
            "Use the fuzzy_search_node tool to find the exact names of entities in the graph.\n"
            "Use get_node_details to get more information about entities.\n"
            "Use find_nodes_by_type_Body to get all the Body type entities mentioned in the schema.\n"
            "Use find_nodes_by_type_OutputComponent to get all the OutputComponent type entities mentioned in the schema.\n"
            "Use get_all_output_relationships to find all PostRequest nodes and their connected nodes in the graph.\n"
            "Use get_all_joint_relationships to get all the data regarding Bodies, joints and connected bodies  in the graph.\n"
            "Refer to the following schema:\n{schema}\n"
            "Based on the user query, use the tools to find relevant entities."
        )
        tool_map = {
            t.name: t for t in self.tools
        }
        chain = prompt | self.llm.bind_tools(self.tools)
        response = chain.invoke({"query": state["query"], "schema": schema})
        new_findings = [f"Task: {state['query']}\n:Results: \n"]
        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_func = tool_map[tool_call["name"]]
                result = tool_func.invoke(tool_call["args"])
                new_findings.append(f"tool_used: {tool_call['name']} \nResult: {result}")
        else:
            new_findings.append(f"\nObservation: {response.content}")

        return {"resolved_entities": new_findings} # Return as message to be handled by ToolNode

    def _cypher_generator_node(self, state: ExecutionStateCypher):
        """Step 2: Use resolved names + schema to write the Cypher."""
        print("--- RESEARCHER: GENERATING CYPHER ---")
        schema = self.neo4j_connector.get_complete_schema_definition()
        
        # Collect findings from tools
        findings = "\n".join(state.get("resolved_entities", []))
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert MBD Cypher developer. Use the schema and MBD Laws below.\n\nSchema:\n{schema}"),
            ("human", "Generate a Cypher query to answer: {query}. Previous  findings from a broken down query:\n{findings}\n\n"
                      "Only return the Cypher query inside ```cypher blocks.")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({
            "query": state["query"],
            "schema": schema,            
            "findings": findings
        })
        
        # Extract query from markdown
        query = response.content.split("```cypher")[-1].split("```")[0].strip()
        return {"cypher_query": query}

    def _query_executor_node(self, state: ExecutionStateCypher):
        """Step 3: Run the query against Neo4j."""
        print(f"--- RESEARCHER: EXECUTING QUERY ---\n{state['cypher_query']}")
        try:
            raw_results = self.neo4j_connector.query(state["cypher_query"])
            formatted = self.neo4j_connector.format_results_to_text(raw_results)
            return {"results": formatted, "iteration_count": state["iteration_count"] + 1}
        except Exception as e:
            return {"errors": [str(e)], "iteration_count": state["iteration_count"] + 1}

    # --- ROUTING ---

    def _check_results(self, state: ExecutionStateCypher):
        """Decide if we need to retry or if results are sufficient."""
        check_prompt_template = " Given the results: {results}, the cypher query that produced them {cypher_query}, do they sufficiently answer the query: {query}? Reply with YES or NO along with a precise reason."
        prompt = ChatPromptTemplate.from_template(check_prompt_template)
        chain = prompt | self.llm
        response = chain.invoke({
            "results": state.get("results", "No results found."),
            "query": state["query"],
            "cypher_query": state.get("cypher_query", "")
        })
        if "YES" in response.content.upper() or self.iterations >=3:
            return END
        else:
            self.iterations += 1
            schema = self.neo4j_connector.get_complete_schema_definition()
            # reframe the query or try again
            check_prompt_template = " Given the results: {results}," \
            " the cypher queries did not sufficiently answer the query: {query}. " \
            " Change the input question using the schema: {schema} and the intermediate findings {findings}. Start the search elsewhere " \
            " Reply with the new query only."
            prompt = ChatPromptTemplate.from_template(check_prompt_template)
            chain = prompt | self.llm
            response = chain.invoke({
                "results": state.get("results", "No results found."),
                "query": state["query"],
                "schema": schema,
                "findings": "\n".join(state.get("resolved_entities", [])[-1])
            })
            state["query"] = response.content.strip()
            return "resolver"
        
    def _create_sub_graph(self) -> StateGraph:
        workflow = StateGraph(ExecutionStateCypher)

        workflow.add_node("resolver", self._entity_resolver_node)        
        workflow.add_node("generator", self._cypher_generator_node)
        workflow.add_node("executor", self._query_executor_node)

        workflow.set_entry_point("resolver")
        
        # Flow: Resolver -> Tools -> Generator -> Executor -> [Check]
        workflow.add_edge("resolver", "generator")
        workflow.add_edge("generator", "executor")
        
        workflow.add_conditional_edges(
            "executor",
            self._check_results,
            {
                END: END,
                "resolver": "resolver"
            }
        )

        return workflow.compile()

    def research_and_cypher_query_result(self, query: str) -> str:
        """Entry point for the ToolBelt/Analyst."""
        self.iterations = 1
        initial_state = {
            "query": query,
            "iteration_count": 0,
            "resolved_entities": [],
            "errors": []
        }
        final_state = self.subgraph.invoke(initial_state)
        
        if final_state.get("results"):
            return f"Cypher Used: {final_state['cypher_query']}\nResults: {final_state['results']}"
        else:
            return f"Failed to find data. Last Query: {final_state.get('cypher_query')}. Errors: {final_state.get('errors')}"