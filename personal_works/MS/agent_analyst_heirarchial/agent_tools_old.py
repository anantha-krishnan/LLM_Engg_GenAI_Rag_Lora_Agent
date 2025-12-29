import json
from neo4j_kg_builder import Neo4jConnector
from thefuzz import process
from langchain_experimental.utilities import PythonREPL
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser

import global_vars
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sys
import io

class ToolBelt:
    """A collection of tools the analyst agent can use to investigate the KG."""
    def __init__(self, connector: Neo4jConnector):
        self.neo4j_connector = connector
        self.repl = PythonREPL()
        self.llm = ChatOpenAI(
            model_name=global_vars.model_openai_4omini,
            #openai_api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
            #openai_api_key=global_vars.google_api_key,
            temperature=0.3,
            streaming=True,
        )

    def run_python_analysis(self, cypher_query: str, python_code: str) -> str:
        """
        1. Executes cypher_query to get time-series/numerical data.
        2. Injects data into a DataFrame 'df'.
        3. Executes complex python_code (SciPy/NumPy/Pandas) for engineering analysis.
        """
        # add colour to terminal prints
        print("\033[94m")  # Set text color to blue
        print(f"\n--- TOOL: run_advanced_analysis (Numerical) ---")
        print(python_code)
        print("\033[0m")  # Reset text color to default
        
        try:
            # 1. Fetch data
            raw_results = self.neo4j_connector.query(cypher_query)
            if not raw_results:
                return "Error: Cypher query returned no data. Check your MATCH statement."
            
            df = pd.DataFrame(raw_results)
            
            # 2. Prepare environment with high-end libraries
            local_vars = {
                "df": df,
                "np": np,
                "pd": pd,
                "sp": sp,       # SciPy Signal (FFTs, Filters)
                "plt": plt             # You can add matplotlib if needed
            }

            # 3. Capture output and execute
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            
            try:
                sys.stdout = stdout_capture
                sys.stderr = stderr_capture
                exec(python_code, globals(), local_vars)
            except Exception as e:
                return f"Python Execution Error: {str(e)}\nTraceback: {stderr_capture.getvalue()}"
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr

            output = stdout_capture.getvalue()
            print(f"--- Analysis Output ---\n{output}")
            return f"Data Analysis Result:\n{output if output else 'Code executed successfully, but nothing was printed.'}"

        except Exception as e:
            return f"General Error: {str(e)}"   
        
    def query_graph_for_context(self, node_types: list[str], node_names: list[str]) -> str:
        """
        Queries the Neo4j graph for details about specific nodes by type or name
        and returns a formatted string of the results.
        """
        if not node_types and not node_names:
            return "No query was executed as no node types or names were specified."

        all_details = []
        if node_types:
            for n_type in node_types:
                # Assuming you have a method like this in Neo4jConnector
                nodes_type = self.neo4j_connector.get_nodes_by_type(n_type)
                if nodes_type:
                    node_names.extend(nodes['name'] for nodes in nodes_type)
        node_names = list(set(node_names))  # Remove duplicates
        if node_names:
            for n_name in node_names:
                # Assuming you have a method like this in Neo4jConnector
                details = self.neo4j_connector.get_dossier_for_any_entity(n_name)
                if details:
                    all_details.append(details)
        
        if not all_details:
            all_details.append(f"No information found in knowledge graph for query: types={node_types}, names={node_names}")

        return all_details, node_names
    
    def find_matching_node_names(self, query_string: str) -> str:
        """
        Executes a fuzzy search for node names against the KG to find the
        most likely starting points for an investigation.
        """
        # add colour to terminal prints
        print("\033[92m")  # Set text color to green
        print(f"\n--- TOOL: find_matching_node_names ---")
        print("\033[0m")  # Reset text color to default
        all_nodes = self.neo4j_connector.get_all_nodes_with_primary_type()
        if not all_nodes:
            return "The Knowledge Graph is empty. No components to search."

        node_name_to_type = {node['name']: node['type'] for node in all_nodes}
        all_node_names = list(node_name_to_type.keys())

        # Use fuzzy matching to find potential candidates
        matches = process.extractBests(query_string, all_node_names, score_cutoff=70, limit=5)

        if not matches:
            return (f"No components found matching '{query_string}'. "
                   f"Try rephrasing or use get_graph_schema to see all component types.")

        formatted_matches = [
            f"- '{name}' (Type: {node_name_to_type[name]})" for name, score in matches
        ]
        result_string = "Found potential component matches:\n" + "\n".join(formatted_matches)
        #print(result_string)
        return result_string
    def find_nodes_by_type(self, node_type: str) -> str:
        """
        Finds all nodes of a given type in the knowledge graph.
        """
        # add colour to terminal prints
        print("\033[93m")  # Set text color to yellow
        print(f"\n--- TOOL: find_nodes_by_type for type '{node_type}' ---")
        print("\033[0m")  # Reset text color to default
        nodes = self.neo4j_connector.get_nodes_by_type(node_type)
        if not nodes:
            return f"No components found of type '{node_type}'."

        formatted_nodes = [
            f"'{node['name']}'" for node in nodes
        ]
        result_string = f"Components of type '{node_type}':\n" + "\n".join(formatted_nodes)
        return result_string
    def get_enriched_dossier(self, entity_name: str) -> str:
        """
        Retrieves a complete, enriched dossier for a single entity, including
        its graph connections and an explanation from official documentation.
        This is the primary tool for gathering deep context.
        """
        print("\033[95m")  # Set text color to purple
        print(f"\n--- TOOL: get_enriched_dossier for '{entity_name}' ---")
        print("\033[0m")  # Reset text color to default
        if not self.neo4j_connector.entity_exists(entity_name):
            return f"Error: The component '{entity_name}' does not exist in the Knowledge Graph."

        # 1. Get the structural information from the KG
        #graph_dossier = self.neo4j_connector.get_dossier_for_any_entity(entity_name)

        # 2. Get the raw properties for the documentation lookup
        raw_node_data = self.neo4j_connector.get_node_properties(entity_name)

        # 3. Call the agent's "auto-researcher" to get the docs explanation
        # We pass the agent instance to access its _get_documentation_explanation method
        doc_explanation = self._get_documentation_explanation(raw_node_data)

        # 4. Combine into a single block
        return "" + "\n" + doc_explanation

    def _get_documentation_explanation(self, node_data: dict) -> str:
        """Your exact _get_documentation_explanation function from the old agent."""
        if not node_data: 
            return ""
        node_type = next((label for label in node_data.get('_labels', []) if label != 'Node'), None)
        node_data.pop('_labels', None)
        if not node_type: 
            return ""
        if (node_type.lower() == "postrequest" and node_data.get("measurement").lower() == 'usersub') or (node_type.lower() == "outputcomponent"):
            node_type = 'pr_usersub'
        term_to_lookup = node_type.lower()
        url_map = {
            "marker": "https://help.altair.com/hwsolvers/ms/topics/solvers/ms/xml-format_90.htm",
            "motion": "https://help.altair.com/hwsolvers/ms/topics/solvers/ms/xml-format_74.htm",
            "joint": "https://help.altair.com/hwsolvers/ms/topics/solvers/ms/xml-format_41.htm",
            "body": "https://help.altair.com/hwsolvers/ms/topics/solvers/ms/xml-format_35.htm",
            "postrequest": "https://help.altair.com/hwsolvers/ms/topics/solvers/ms/xml-format_83.htm",
            "pr_usersub": "https://help.altair.com/hwdesktop/hwx/topics/motionview/coordinate_systems_and_output_request_r.htm",
            "autotiresystem": "https://help.altair.com/hwdesktop/hwx/topics/motionview/fiala_tire_force_calculation_r.htm",
            "stateequation": "https://help.altair.com/hwsolvers/ms/topics/solvers/ms/gsesub_gsexx_gsexu_gseyx_gseyu.htm",
            "force": "https://help.altair.com/hwsolvers/ms/topics/solvers/ms/gfosub.htm"
        }
        url_to_load = url_map.get(term_to_lookup)
        if not url_to_load: 
            return ""

        try:
            print(f"  -> Auto-researching type '{node_type}' for entity '{node_data.get('name')}'...")
            loader = WebBaseLoader([url_to_load])
            docs = loader.load()
            raw_content = "\n".join([doc.page_content for doc in docs])
            synthesis_prompt = ChatPromptTemplate.from_template(
                """You are a helpful assistant. Explain a piece of simulation data using the provided documentation.
                **Documentation for a '{term}' component:**\n{documentation}\n
                **Data from User's Component:**\n{data_context}\n
                **Task:** Based ONLY on the documentation, concisely explain the key properties/values from the user's data."""
            )
            synthesis_chain = synthesis_prompt | self.llm | StrOutputParser()
            explanation = synthesis_chain.invoke({
                "documentation": raw_content, 
                "data_context": str(node_data), 
                "term": node_type
            })
            return f"\n--- Official Documentation Explanation for this {node_type} ---\n{explanation}"
        except Exception as e:
            print(f"  -> ERROR during documentation enrichment for '{node_type}': {e}")
            return ""