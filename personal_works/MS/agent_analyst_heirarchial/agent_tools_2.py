import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import io
from contextlib import redirect_stdout
import sys
from thefuzz import process
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

class ToolBelt:
    def __init__(self, connector,llm):
        self.neo4j_connector = connector
        self.llm = llm

    def run_python_analysis(self, cypher_query: str, python_code: str) -> str:
        """Executes Cypher to get data, then runs engineering Python code."""
        try:
            print(f"\033[93m--- TOOL: run_python_analysis ' ---\033[0m")
            raw_results = self.neo4j_connector.query(cypher_query)
            if not raw_results:
                return "Error: Cypher query returned no data."
            
            df = pd.DataFrame(raw_results[0].data())
            safe_globals = {"__builtins__": __builtins__} 
            local_scope = {"df": df, "np": np, "pd": pd, "sp": sp}
            stdout_capture = io.StringIO()
            
            try:
                with redirect_stdout(stdout_capture):
                    exec(python_code, safe_globals, local_scope)
                
                # This is your final analysis string!
                analysis_result = stdout_capture.getvalue()

            except Exception as e:
                analysis_result = f"Simulation analysis failed with error: {str(e)}"

            return f"Analysis Result:\n{analysis_result}"
        except Exception as e:
            return f"Error: {str(e)}"

    def get_post_request_components_map(self, pr) -> str:
        """Lists all postrequest to OutputComponents mapping."""        
        query = """
        MATCH (pr:PostRequest {name:$pr})-[:HAS_COMPONENT]->(oc:OutputComponent)
        RETURN pr.name as postrequest, collect(oc.name) as components        
        """
        records = self.neo4j_connector.query(query, {"pr": pr})
        if not records:
            return "No PostRequest to OutputComponent mappings found."
        
        lines = []
        for r in records:
            comps = ", ".join(r['components'])
            # lines.append(f"- {r['postrequest']} has the following OutputComponents: {comps}")
        return comps 
    
    def fuzzy_search_node(self, query_string: str, full_task: str) -> dict:
        """Finds the real name of a node in the KG."""
        print(f"\033[95m--- TOOL: fuzzy_search_node ' ---\033[0m")
        all_nodes = self.neo4j_connector.get_all_nodes_with_primary_type()
        # store a map of lower case to actual names of nodes
        names = {n['name'].lower():n['name'] for n in all_nodes}        
        matches = process.extractBests(query_string.lower(), names.keys(), score_cutoff=70, limit=3)
        # get the value back from the original names map        
        matches = [names[n[0]] for n in matches]
        if not matches: return {"error": "No nodes found"}
        
        connections = self.get_node_neighbors(matches[0]) 
        node_details = self.get_enriched_dossier(matches[0])
        return {"matches": results}
        results = []
        for name, score in matches:
            node_type = next(n['type'] for n in all_nodes if n['name'] == name)
            results.append({"name": name, "type": node_type})

    def get_node_neighbors(self, node_name: str) -> str:
        """Explores connections: (node)-[rel]->(neighbor)"""
        query = """
        MATCH (n {name: $name})-[r]-(m)
        RETURN labels(m)[0] as type, m.name as name, type(r) as rel
        LIMIT 15
        """
        print(f"\033[96m--- TOOL: get_node_neighbors ' ---\033[0m")
        records = self.neo4j_connector.query(query, {"name": node_name})
        if not records: return f"No neighbors found for {node_name}"
        
        lines = [f"- {r['rel']} -> {r['name']} ({r['type']})" for r in records]
        return f"Connections for {node_name}:\n" + "\n".join(lines)
    
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

    def fetch_numerical_data(self, node_name: str, full_task: str=None) -> str:
        """
        Smart data fetcher. 
        1. If target is an OutputComponent: Returns data statistics.
        2. If target is a PostRequest: Returns a list of available OutputComponents and their directions (FX, FY, etc.).
        """
        print(f"\033[94m--- TOOL: fetch_numerical_data for '{node_name}' ---\033[0m")
        # 1. First, check what kind of node we are dealing with and if it has data
        query = """
        MATCH (n {name: $name})
        RETURN labels(n) as labels, n.time_values as time, n.output_values as val, n.type as type
        """
        res = self.neo4j_connector.query(query, {"name": node_name})
        
        if not res:
            return f"Error: No node named '{node_name}' found in the graph."

        node_info = res[0]
        labels = node_info['labels']
        selected_component = None
        # If it's a PostRequest, we may need to auto-select an OutputComponent based on
        if "PostRequest" in labels:
            pr_oc_map = self.get_post_request_components_map(node_name)
            synthesis_prompt = ChatPromptTemplate.from_template(
                """You are a technical selector. 
                TASK: {task}
                POSTREQUEST: {pr}
                AVAILABLE OUTPUT COMPONENTS: {components_map}

                INSTRUCTION:
                Which specific OutputComponent name from the list of OUTPUT COMPONENTS is required to complete the TASK? 
                - Pick the keywords in task OTHER THAN POSTREQUEST.
                - Match the keywords to the list of OUTPUT COMPONENTS.
                - Return ONLY the single best-matching OUTPUT COMPONENT name.
                - If no specific component can be identified, return 'NONE'.
                - Keywords are usually inside quotes
                
                RESULT: Provide ONLY the single OUTPUT COMPONENT name or 'NONE'."""
            )
            synthesis_chain = synthesis_prompt | self.llm | StrOutputParser()
            selected_component = synthesis_chain.invoke({
                "pr": node_name,
                "task": full_task,
                "components_map": str(pr_oc_map)
            }).strip().replace("'", "").replace("\"", "")
            
                
        

        # --- CASE A: The node is a PostRequest (The container) ---
        if "PostRequest" in labels and selected_component is None:
            print(f"  -> Node is a PostRequest. Fetching child components...")
            comp_query = """
            MATCH (n {name: $name})-[:HAS_COMPONENT]->(oc:OutputComponent)
            RETURN oc.name as name, oc.type as type
            """
            components = self.neo4j_connector.query(comp_query, {"name": node_name})
            
            if not components:
                return f"The PostRequest '{node_name}' was found, but it has no connected OutputComponent nodes (no data). User did not update the data entry into the KG."
            
            comp_list = [f"- {c['name']} (Type: {c['type']})" for c in components]
            return (f"'{node_name}' is a PostRequest containing the following data components:\n" + 
                    "\n".join(comp_list) + 
                    "\n\nACTION: Please call 'fetch_numerical_data' again using one of the specific component names above.")
        elif "PostRequest" in labels and selected_component != "NONE":
            if selected_component in str(pr_oc_map) and selected_component != "NONE":
                query = """
                MATCH (pr:PostRequest {name:$pr})-[:HAS_COMPONENT]->(oc:OutputComponent {name: $selected_component}) RETURN oc.output_values as val, oc.time_values as time
                """
                print(f"  -> Auto-resolved to child component: {selected_component}")
                # Recursively call the same function with the CHILD name to get the actual data
                components = self.neo4j_connector.query(query, {"pr": node_name, "selected_component": selected_component})
                if components:
                    node_info['val'] = components[0]['val']
                    node_info['time'] = components[0]['time']
        # --- CASE B: The node is an OutputComponent (The actual data) ---
        if node_info['time'] is not None and len(node_info['time']) > 0:
            time_data = node_info['time']
            val_data = node_info['val']
            
            stats = {
                "count": len(time_data),
                "min": min(val_data),
                "max": max(val_data),
                "mean": sum(val_data) / len(val_data)
            }
            
            return (f"SUCCESS: Numerical data retrieved for OutputComponent '{node_name}'.\n"
                    f"- Data Points: {stats['count']}\n"
                    f"- Range: [{stats['min']:.4f}, {stats['max']:.4f}]\n"
                    f"- Mean: {stats['mean']:.4f}\n"
                    f"You may now proceed to 'python_analysis' using this node name {node_name} in your Cypher query.")

        # --- CASE C: It's something else (e.g., a Body) ---
        return (f"Node '{node_name}' (Type: {labels}) does not contain numerical arrays. "
                f"If this is a Body, use 'get_neighbors' to reach the nearest PostRequest measuring it.")

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
                """You are a Altair Motion Solve MBD model analyst assistant. Explain a piece of simulation data using the provided documentation.
                **Documentation for '{term}' component:**\n{documentation}\n
                **Data defined for '{term}' in the User's MBD Model :**\n{data_context}\n
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
        
    def list_nodes_by_type(self, label: str) -> str:
        """Returns all names of a specific node type. Useful when fuzzy search fails."""
        print(f"\033[97m--- TOOL: list_nodes_by_type '{label}' ---\033[0m")
        try:
            query = f"MATCH (n:{label}) RETURN n.name as name LIMIT 50"
            records = self.neo4j_connector.query(query)
            names = [r['name'] for r in records]
            return f"All {label} nodes in graph: " + ", ".join(names)
        except Exception as e:
            return f"Error: label '{label}' is not a valid node type"
    def list_all_nodes(self) -> str:
        """Returns all names of all nodes in the KG."""
        print(f"\033[97m--- TOOL: list_all_nodes ---\033[0m")
        try:
            query = f"MATCH (n) RETURN n.name as name"
            records = self.neo4j_connector.query(query)
            names = [r['name'] for r in records]
            return f"All nodes in graph: " + ", ".join(names)
        except Exception as e:
            return f"Error: {str(e)}"