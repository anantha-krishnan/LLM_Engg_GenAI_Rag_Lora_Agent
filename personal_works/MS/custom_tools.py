# custom_tools.py (NEW FILE)

from langchain_core.tools import BaseTool
from typing import Type
from pydantic.v1 import BaseModel, Field # Use pydantic v1 for BaseTool compatibility

# --- Import your existing helper classes ---
# We still use them to hold the core logic
from crew_agent_tools import ArchivistToolbelt, KGNavigatorToolbelt

# --- Define the input schema for our tools ---
# This tells the tool what arguments to expect
class SearchInput(BaseModel):
    query: str = Field(description="The search query for finding documentation or test cases.")

class DossierInput(BaseModel):
    entity_name: str = Field(description="The exact name of the component to investigate in the Knowledge Graph.")

# --- Create the explicit Tool Classes ---

class SearchDocumentationTool(BaseTool):
    """A tool to search for relevant test cases, models, or documentation."""
    name: str = "Search Documentation"
    description: str = "Use this tool to search the vector store for relevant test cases, models, or documentation based on a user query."
    args_schema: Type[BaseModel] = SearchInput
    
    # We will instantiate our helper class when this tool is created
    _toolbelt: ArchivistToolbelt

    @classmethod
    def from_retriever(cls, retriever):
        """A factory method to properly initialize the tool with its dependency."""
        return cls(_toolbelt=ArchivistToolbelt(retriever=retriever))

    def _run(self, query: str):
        """Use the tool."""
        return self._toolbelt.search_documentation(query=query)


class GetEnrichedDossierTool(BaseTool):
    """A tool to perform a deep-dive analysis on a specific model component."""
    name: str = "Get Enriched Component Dossier"
    description: str = "Use this tool to perform a deep-dive investigation on a specific component (entity) in the Knowledge Graph. It retrieves its connections, properties, and enriches this with explanations from official documentation."
    args_schema: Type[BaseModel] = DossierInput
    
    _toolbelt: KGNavigatorToolbelt

    @classmethod
    def from_dependencies(cls, neo4j_connector, llm):
        """A factory method to properly initialize the tool with its dependencies."""
        return cls(_toolbelt=KGNavigatorToolbelt(neo4j_connector=neo4j_connector, llm=llm))

    def _run(self, entity_name: str):
        """Use the tool."""
        return self._toolbelt.get_enriched_dossier(entity_name=entity_name)