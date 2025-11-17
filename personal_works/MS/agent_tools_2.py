import json
from neo4j_kg_builder import Neo4jConnector
from thefuzz import process


class ToolBelt:
    """A collection of tools the analyst agent can use to investigate the KG."""
    def __init__(self, connector: Neo4jConnector):
        self.neo4j_connector = connector

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
        print(f"\n--- TOOL: find_matching_node_names ---")
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
        print(result_string)
        return result_string
    def find_nodes_by_type(self, node_type: str) -> str:
        """
        Finds all nodes of a given type in the knowledge graph.
        """
        print(f"\n--- TOOL: find_nodes_by_type for type '{node_type}' ---")
        nodes = self.neo4j_connector.get_nodes_by_type(node_type)
        if not nodes:
            return f"No components found of type '{node_type}'."

        formatted_nodes = [
            f"- '{node['name']}' (ID: {node['id']})" for node in nodes
        ]
        result_string = f"Components of type '{node_type}':\n" + "\n".join(formatted_nodes)
        print(result_string)
        return result_string
    def get_enriched_dossier(self, entity_name: str, qa_agent_instance) -> str:
        """
        Retrieves a complete, enriched dossier for a single entity, including
        its graph connections and an explanation from official documentation.
        This is the primary tool for gathering deep context.
        """
        print(f"\n--- TOOL: get_enriched_dossier for '{entity_name}' ---")
        if not self.neo4j_connector.entity_exists(entity_name):
            return f"Error: The component '{entity_name}' does not exist in the Knowledge Graph."

        # 1. Get the structural information from the KG
        graph_dossier = self.neo4j_connector.get_dossier_for_any_entity(entity_name)

        # 2. Get the raw properties for the documentation lookup
        raw_node_data = self.neo4j_connector.get_node_properties(entity_name)

        # 3. Call the agent's "auto-researcher" to get the docs explanation
        # We pass the agent instance to access its _get_documentation_explanation method
        doc_explanation = qa_agent_instance._get_documentation_explanation(raw_node_data)

        # 4. Combine into a single block
        return graph_dossier + "\n" + doc_explanation

    def get_graph_schema(self) -> str:
        """Returns the high-level schema of the knowledge graph."""
        print(f"\n--- TOOL: get_graph_schema ---")
        return self.neo4j_connector.get_graph_schema()
   