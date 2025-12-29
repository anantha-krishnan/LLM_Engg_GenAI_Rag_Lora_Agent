# crew_analyst_agent.py

from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Import our tools and setup functions ---
from crew_agent_tools import (
    GetGraphSchemaTool,
    FindMatchingNodeNamesTool,
    GetEnrichedDossierTool
)
import global_vars
from ingest_ms_tests import factory_create_vector_store, factory_get_hybrid_retriever

class CrewAIAnalystAgent:
    def __init__(self, llm_provider="openai"):
        if llm_provider == "google":
            print("--- Using Google Gemini 1.5 Pro ---")
            self.manager_llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")
            self.worker_llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")
        else:  # Default to OpenAI
            print("--- Using OpenAI GPT-4o ---")
            self.manager_llm = ChatOpenAI(model_name=global_vars.model_openai_4o, temperature=0.3)
            self.worker_llm = ChatOpenAI(model_name=global_vars.model_openai_4omini, temperature=0.3)
            
        # --- Initialize Toolbelts ---
        METADATA_CSV = "MS_Tests_Metadata.csv"
        vs = factory_create_vector_store(metadata_csv_path=METADATA_CSV, vector_store_type="chroma")
        retriever = factory_get_hybrid_retriever(vs, alpha=0.5, top_k=5)

        # --- Define Worker Agents (NOT the manager) ---
        # IMPORTANT: Use simple, unique role names that are easily identifiable
        
        self.component_name_resolver = Agent(
            role='Component Name Resolver',
            goal='Find the exact, official name of any model component based on a user\'s ambiguous or partial query.',
            backstory=(
                'You are a meticulous librarian for the engineering knowledge graph. '
                'Your sole function is to use the `FindMatchingNodeNamesTool` to resolve ambiguity. '
                'You take a fuzzy query and return a list of precise, matching component names.'
            ),
            llm=self.worker_llm,
            verbose=True,
            tools=[FindMatchingNodeNamesTool],
            allow_delegation=False  # Worker agents should not delegate
        )

        self.data_dossier_analyst = Agent(
            role='Data Dossier Analyst',
            goal='Retrieve a complete and detailed dossier of information for a component with a known, exact name.',
            backstory=(
                'You are a deep-dive data analyst. When given an exact component name, you use the '
                '`GetEnrichedDossierTool` to pull all available data: its properties, its connections to other components, '
                'any associated numerical results, and explanations from official documentation.'
            ),
            llm=self.worker_llm,
            verbose=True,
            tools=[GetEnrichedDossierTool],
            allow_delegation=False
        )
        
        self.graph_schema_specialist = Agent(
            role='Graph Schema Specialist',
            goal='Provide a high-level map of the entire knowledge graph structure.',
            backstory=(
                'You are a database architect. You do not look at individual data points. '
                'Your only job is to use the `GetGraphSchemaTool` to report on the types of nodes '
                'and the relationships between them. This helps the manager understand how different parts of the model are connected.'
            ),
            llm=self.worker_llm,
            verbose=True,
            tools=[GetGraphSchemaTool],
            allow_delegation=False
        )

        # --- Define Manager Agent ---
        # The manager should be created but will be passed to Crew separately
        self.research_manager = Agent(
            role='CAE Research Manager',
            goal='Orchestrate a team of specialists to answer complex questions about MotionSolve models by delegating tasks to specialist agents.',
            backstory=(
                'You are an expert engineering manager. You do not perform technical work yourself. '
                'Instead, you analyze a user\'s request and delegate tasks to your specialist agents. '
                'You have three specialists available:\n'
                '1. Component Name Resolver - finds exact component names\n'
                '2. Data Dossier Analyst - retrieves detailed component information\n'
                '3. Graph Schema Specialist - provides graph structure overview\n\n'
                'Your job is to create a plan, delegate tasks in the right sequence, and synthesize the final report.'
            ),
            llm=self.manager_llm,
            verbose=True,
            allow_delegation=True  # Manager must be able to delegate
        )

    def _reformulate_question(self, message: str, chat_history: list[BaseMessage]) -> str:
        """Handles the history-aware reformulation of the user's question."""
        if not chat_history:
            return message

        print("\n---REFORMULATING QUESTION WITH HISTORY---")
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        chain = contextualize_q_prompt | self.manager_llm | StrOutputParser()
        standalone_question = chain.invoke({"chat_history": chat_history, "question": message})
        print(f"---Standalone Question: {standalone_question}---")
        return standalone_question

    def process_message(self, message: str, chat_history: list[BaseMessage]):
        """
        The main entry point for the Gradio UI.
        Handles question reformulation and runs the research crew.
        """
        standalone_question = self._reformulate_question(message, chat_history)

        # Create a task for the manager
        research_task = Task(
            description=(
                f"Answer this user question: '{standalone_question}'.\n\n"
                "You must delegate to your specialist team to gather information:\n\n"
                "**Available Specialists:**\n"
                "- Component Name Resolver: Finds exact, official component names from fuzzy queries\n"
                "- Data Dossier Analyst: Retrieves detailed information for components with known exact names\n"
                "- Graph Schema Specialist: Provides overview of the knowledge graph structure\n\n"
                "**Your Process:**\n"
                "1. Analyze the question and determine which specialists you need\n"
                "2. Delegate tasks to specialists in logical sequence (typically: resolve names first, then get detailed data)\n"
                "3. Review the information returned by your specialists\n"
                "4. Synthesize a comprehensive answer that directly addresses the user's question\n"
                "5. Present your findings in a clear, well-structured report\n\n"
                "Work efficiently - aim to complete in 3-4 delegation cycles."
            ),
            expected_output=(
                "A comprehensive, well-structured report that answers the user's question based on "
                "information gathered from the specialist agents. The report should explain findings "
                "in clear engineering language, not just raw data dumps."
            ),
            agent=self.research_manager
        )

        # --- Assemble and Run the Crew ---
        # CRITICAL: Only worker agents go in the agents list
        # The manager is specified separately via manager_agent parameter
        research_crew = Crew(
            agents=[
                self.component_name_resolver,
                self.data_dossier_analyst,
                self.graph_schema_specialist
            ],
            tasks=[research_task],
            process=Process.hierarchical,
            manager_agent=self.research_manager,
            verbose=True,
            # memory=False  # Set to True if you want crew-level memory
        )
        # ============ DEBUG: LIST COWORKERS ============
        print("\n" + "="*60)
        print("COWORKERS DEBUG:")

        # Check if the manager has a coworkers attribute
        if hasattr(self.research_manager, '_coworkers'):
            print(f"Manager's coworkers (_coworkers): {self.research_manager._coworkers}")

        # Check the crew's agents
        print(f"\nCrew agents: {[agent.role for agent in research_crew.agents]}")

        # Try to access the manager's coworkers after crew initialization
        if hasattr(research_crew, 'manager_agent') and research_crew.manager_agent:
            manager = research_crew.manager_agent
            print(f"\nManager role: {manager.role}")
            
            # Different CrewAI versions store coworkers differently
            for attr in ['_coworkers', 'coworkers', 'agents', '_agents']:
                if hasattr(manager, attr):
                    print(f"Manager.{attr}: {getattr(manager, attr)}")

        print("="*60 + "\n")
        # ================================================
        try:
            final_result = research_crew.kickoff()
            yield str(final_result)
        except Exception as e:
            yield f"Error during crew execution: {str(e)}"

    def close(self):
        """Cleanly close any open connections."""
        if hasattr(self, 'neo4j_connector'):
            self.neo4j_connector.close()