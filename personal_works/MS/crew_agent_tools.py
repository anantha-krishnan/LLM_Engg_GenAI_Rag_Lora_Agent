# crew_agent_tools.py
#from crewai.tools import tool
from langchain_core.tools import tool  # Use LangChain's tool decorator, NOT CrewAI's
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from thefuzz import process

from neo4j_kg_builder import Neo4jConnector
import global_vars

# --- Toolbelt for the Archivist Agent (RAG Specialist) ---
class ArchivistToolbelt:
    def __init__(self, retriever):
        self.retriever = retriever

    def _format_docs_as_context(self, docs):
        """Your exact _format_docs_as_context function from the old agent."""
        formatted_docs = []
        for i, doc in enumerate(docs):
            meta = doc.metadata
            doc_str = (
                f"--- Test Model {i+1} ---\n"
                f"Description: {doc.page_content}\n"
                f"working_dir: {meta.get('working_dir', 'N/A')}\n"
                f"main_category: {meta.get('main_category', 'N/A')}\n"
                f"sub_category_event: {meta.get('sub_category_event', 'N/A')}\n"
                f"model_name: {meta.get('model_name', 'N/A')}\n"
                f"export_xml: {meta.get('export_xml', 'N/A')}\n"
                f"sub_folder: {meta.get('sub_folder', 'N/A')}\n"
                f"tags: {meta.get('tags', 'N/A')}\n"
                f"plot_reports: {meta.get('plot_reports', 'N/A')}\n"
            )
            formatted_docs.append(doc_str)
        return "\n\n".join(formatted_docs)

    @tool
    def search_documentation(self, query: str) -> str:
        """
        Searches the vector store for relevant test cases, models, or documentation
        based on a user query. Returns a formatted string of findings.
        """
        print(f"\n---TOOL: ARCHIVIST searching for: '{query}'---")
        docs = self.retriever.invoke(query)
        if not docs:
            return "No relevant documents found for the query: " + query
        return self._format_docs_as_context(docs)


# --- Toolbelt for the KG Navigator Agent (Model & Results Specialist) ---
class KGNavigatorToolbelt:
    def __init__(self, llm: ChatOpenAI=None, neo4j_connector: Neo4jConnector=None):
        if neo4j_connector:
            self.neo4j_connector = neo4j_connector
        else:
            self.neo4j_connector = Neo4jConnector(
                global_vars.NEO4J_URI, 
                global_vars.NEO4J_USER, 
                global_vars.NEO4J_PASSWORD
            )
        if llm:
            self.llm = llm
        else:
            self.llm = ChatOpenAI(
                model_name=global_vars.model_openai_4omini, 
                temperature=0.2
            )

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

    def get_enriched_dossier(self, entity_name: str) -> str:
        """
        Performs a deep-dive investigation on a specific component (entity) in the
        Knowledge Graph. Retrieves connections, properties, and enriches this with
        explanations from official documentation.
        """
        print(f"\n---TOOL: KG NAVIGATOR investigating: '{entity_name}'---")
        if not self.neo4j_connector.entity_exists(entity_name):
            available_components = self.neo4j_connector.get_all_nodes_with_primary_type()
            formatted_components = "\n".join([
                f"- {comp['name']} (Type: {comp['type']})" 
                for comp in available_components
            ])
            return (f"Error: The component '{entity_name}' does not exist in the Knowledge Graph. "
                   f"Please check the name and try again. The available entities in the knowledge graph "
                   f"and its type are:\n{formatted_components}")
        
        graph_dossier = self.neo4j_connector.get_dossier_for_any_entity(entity_name)
        raw_node_data = self.neo4j_connector.get_node_properties(entity_name)
        doc_explanation = self._get_documentation_explanation(raw_node_data)
        
        if not graph_dossier and not doc_explanation:
            return f"No information could be found for '{entity_name}'."
        
        return graph_dossier + doc_explanation

    def find_matching_node_names(self, query_string: str) -> str:
        """Executes a fuzzy search for node names against the KG."""
        print(f"\n--- TOOL LOGIC: Running find_matching_node_names with query: '{query_string}' ---")
        
        all_nodes = self.neo4j_connector.get_all_nodes_with_primary_type()
        if not all_nodes:
            return "The Knowledge Graph appears to be empty. No components are available to search."
            
        node_name_to_type = {node['name']: node['type'] for node in all_nodes}
        all_node_names = list(node_name_to_type.keys())
        
        matches = process.extractBests(query_string, all_node_names, score_cutoff=75, limit=5)
        
        if not matches:
            return (f"No components found in the Knowledge Graph closely matching '{query_string}'. "
                   "Try a different name or use the `get_graph_schema` tool to see what types of components exist.")

        formatted_matches = [
            f"- '{name}' (Type: {node_name_to_type[name]})" for name, score in matches
        ]
        
        result_string = "Found potential component matches in the Knowledge Graph:\n" + "\n".join(formatted_matches)
        print(result_string)
        return result_string

    def get_graph_schema(self) -> str:
        """Retrieves the high-level schema of the graph."""
        print(f"\n--- TOOL LOGIC: Running get_graph_schema ---")
        return self.neo4j_connector.get_graph_schema()


# ============================================================================
# LANGGRAPH-COMPATIBLE TOOLS
# These are standalone functions that LangGraph can use directly
# ============================================================================

# Initialize a global instance for the tools to use
# This will be set up when your main agent initializes
_kg_toolbelt = None

def initialize_kg_toolbelt(neo4j_connector=None, llm=None):
    """Call this from your main agent's __init__ to set up the toolbelt"""
    global _kg_toolbelt
    _kg_toolbelt = KGNavigatorToolbelt(llm=llm, neo4j_connector=neo4j_connector)


@tool
def find_matching_node_names(query_string: str) -> str:
    """
    Find the exact, full name of an entity in the Knowledge Graph based on a potentially 
    incomplete or ambiguous query. Returns a list of the best possible matches.
    The entity can be of various types like Body, Joint, or PostRequest.
    
    Args:
        query_string: The query string to search for matching node names in knowledge graph
        
    Returns:
        A formatted string listing the best matching node names and their types
    """
    if _kg_toolbelt is None:
        raise RuntimeError("KG Toolbelt not initialized. Call initialize_kg_toolbelt() first.")
    return _kg_toolbelt.find_matching_node_names(query_string)


@tool
def get_enriched_dossier(entity_name: str) -> str:
    """
    Perform a deep-dive investigation on a specific component (entity) in the Knowledge Graph.
    Use this AFTER you have the EXACT name of a component. It retrieves the component's 
    properties, connections, and enriches this with explanations from official documentation.
    
    Args:
        entity_name: The exact name of the component to investigate in the Knowledge Graph
        
    Returns:
        A detailed dossier about the component, including its properties, connections, 
        and documentation explanations
    """
    if _kg_toolbelt is None:
        raise RuntimeError("KG Toolbelt not initialized. Call initialize_kg_toolbelt() first.")
    return _kg_toolbelt.get_enriched_dossier(entity_name)


@tool
def get_graph_schema() -> str:
    """
    Understand the overall structure of the model's knowledge graph. Returns a list of all 
    entity types (e.g., Body, Joint) and the relationships that connect them.
    This is extremely useful for creating better research plans for complex questions.
    
    Returns:
        A formatted string describing the graph schema
    """
    if _kg_toolbelt is None:
        raise RuntimeError("KG Toolbelt not initialized. Call initialize_kg_toolbelt() first.")
    return _kg_toolbelt.get_graph_schema()