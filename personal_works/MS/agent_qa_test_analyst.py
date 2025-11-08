# agent_qa_test_analyst.py
from pathlib import Path
from typing import List, TypedDict, Generator, Optional, Any
from operator import itemgetter
import re

import global_vars
from ingest_ms_tests import factory_create_vector_store, factory_get_hybrid_retriever
from neo4j_kg_builder import Neo4jConnector

from langchain_core.messages import BaseMessage
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain.docstore.document import Document
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.document_loaders import WebBaseLoader

from langgraph.graph import StateGraph, END


# --- AGENT SETUP ---
METADATA_CSV = (Path(__file__).parent / "MS_Tests_Metadata.csv").as_posix()
LLM_MODEL_NAME = global_vars.model_openai_4omini

class GraphState(TypedDict):
    # Original fields
    route_decision: str
    question: str
    standalone_question: str # Replaces retrieved_context for holding the question
    chat_history: Optional[List[BaseMessage]]
    documents: List[Document]
    message: List[BaseMessage]
    answer: str
    knowledge_graph: Optional[Any]
    kg_context: str

    # --- ANALYSIS LOOP ---
    entities_to_query: Optional[List[str]]
    queried_entities: set
    judge_decision: str

class DossierReviewOutput(BaseModel):
    next_entities: List[str] = Field(..., description="A list of NEW component names that require their own detailed dossier. Should be an empty list if the current dossier is sufficient.")

class QAAnalystAgent:
    def __init__(self):
        vs = factory_create_vector_store(
            metadata_csv_path=METADATA_CSV,
            vector_store_type="chroma"
        )
        self.retriever = factory_get_hybrid_retriever(vs, alpha=0.5, top_k=500)
        self.llm = ChatOpenAI(
            model_name=LLM_MODEL_NAME,
            temperature=0.3,
            streaming=True,
        )
        self.neo4j_connector = Neo4jConnector(
            global_vars.NEO4J_URI, global_vars.NEO4J_USER, global_vars.NEO4J_PASSWORD
        )
        self.qa_graph = self._create_graph()
    
    def close(self):
        self.neo4j_connector.close()

    # ===================================================================
    # SECTION 1: ROUTING AND BRANCHES
    # ===================================================================

    def route_question(self, state: GraphState) -> str:
        """ Route the question to the appropriate processing path. """
        print("\n---NODE: ROUTING QUESTION---")
        question = state["standalone_question"].lower()
        routing_prompt_str = """You are an expert in routing user queries for a CAE simulation expert system.
                 Based on the user's question, determine which of the three paths they should follow:
                 
                 1. 'rag_branch': The user is searching for a test case, model, or documentation.
                 2. 'kg_structural_branch': The user is asking about the structure or components of a *specific, named* model. The answer does not require looking at numerical result data.
                 3. 'kg_analysis_branch': The user is asking for an analysis or interpretation of simulation *results*. This requires looking at the time-series data and often includes words like "why", "analyze", "converging", "oscillating", "peak".

                 Respond with a JSON object containing the key "branch" with one of the three values: "rag_branch", "kg_structural_branch", or "kg_analysis_branch".
                 """
        routing_prompt = ChatPromptTemplate.from_messages([("system", routing_prompt_str), ("user", "{question}")])
        routing_chain = routing_prompt | self.llm | JsonOutputParser()
        route = routing_chain.invoke({"question": question})
        route_answer = route.get("branch")
        print(f"---INFO: Routing decision: {route_answer}---")
        return {"route_decision": route_answer}

    # ===================================================================
    # SECTION 2: RAG BRANCH NODES
    # ===================================================================

    def _history_aware_retrieval(self, state: GraphState) -> dict:
        """ Reformulate question based on chat history. """
        print("\n---NODE (RAG): HISTORY-AWARE RETRIEVAL---")
        # ... (Your existing code for this function is perfect)
        question = state["question"]
        chat_history = state.get("chat_history", [])
        contextualize_q_system_prompt = """Given a chat history and the latest user question which might reference context
          in the chat history, formulate a standalone question which can be understood 
          without the chat history. Do NOT answer the question, just reformulate it if needed
          and otherwise return it as is. Carefully consider if the user is referring to any particular item in the chat history. Pick the names of those entities."""
        contextualize_q_prompt = ChatPromptTemplate.from_messages([("system", contextualize_q_system_prompt), MessagesPlaceholder(variable_name="chat_history"), ("human", "{question}")])
        history_retriever = contextualize_q_prompt | self.llm | StrOutputParser()
        retrieved_context = history_retriever.invoke({"chat_history": chat_history, "question": question})
        return {"standalone_question": retrieved_context or question}

    def _retrieve_documents(self, state: GraphState) -> dict:
        """ Retrieve relevant documents from vector store. """
        print("\n---NODE (RAG): RETRIEVING DOCUMENTS---")
        question = state["standalone_question"]
        docs = self.retriever.invoke(question)
        return {"documents": docs}

    def _prepare_rag_for_generation(self, state: GraphState) -> dict:
        """ Prepare the context and question for the RAG LLM generation. """
        print("\n---NODE (RAG): PREPARING FOR GENERATION---")
        # ... (Your existing code for this function is perfect)
        context = self._format_docs_as_context(state["documents"])
        question = state["question"]
        qa_sys_prompt = "You are an expert Motion View QA Analyst. Use the following context to answer the question. If you don't know the answer, just say that you don't know. Be concise."
        qa_prompt = ChatPromptTemplate.from_messages([("system", qa_sys_prompt), ("user", "Context:\n{context}\n\nQuestion: {question}")])
        messages = qa_prompt.invoke({"context": context, "question": question})
        return {"message": messages.to_messages()}

    # ===================================================================
    # SECTION 3: KG STRUCTURAL BRANCH NODES
    # ===================================================================

    def _query_kg_for_structure(self, state: GraphState) -> dict:
        """ Queries Neo4j for structural context. """
        print("\n---NODE (KG Structure): QUERYING NEO4J---")
        question = state["standalone_question"]
        entities = re.findall(r"'([^']*)'|\"([^\"]*)\"", question)
        entities = [name for tpl in entities for name in tpl if name]

        if not entities:
            context = "The user's question did not specify a component name. Please ask them to specify a component, e.g., 'Describe \"Ground Body\"'."
            return {"kg_context": context}

        print(f"---INFO: Extracted entities: {entities}---")
        all_results = []
        for entity_name in entities:
            cypher_query = "MATCH (n {name: $name}) OPTIONAL MATCH (n)-[r]-(neighbor) RETURN n, r, neighbor"
            results = self.neo4j_connector.query(cypher_query, parameters={"name": entity_name})
            all_results.extend(results)

        kg_context = self.neo4j_connector.format_results_to_text(all_results)
        print(f"---INFO: Extracted Structural KG Context:\n{kg_context}---")
        return {"kg_context": kg_context}

    def _prepare_kg_structure_for_generation(self, state: GraphState) -> dict:
        """ Prepares the prompt for structural questions. """
        print("\n---NODE (KG Structure): PREPARING FOR GENERATION---")
        context = state["kg_context"]
        question = state["question"]
        sys_prompt = """You are an expert MotionSolve CAE Analyst. 
        You have been provided with context from a model's Knowledge Graph.
        Use ONLY this provided context to answer the user's question about the model's structure.
        If the context doesn't contain the answer, say that.
        """
        kg_prompt = ChatPromptTemplate.from_messages([("system", sys_prompt), ("user", "Context:\n{context}\n\nQuestion: {question}")])
        messages = kg_prompt.invoke({"context": context, "question": question})
        return {"message": messages.to_messages()}

    # ===================================================================
    # SECTION 4: KG ANALYSIS BRANCH (LOOP) NODES
    # ===================================================================

    def _query_kg_for_initial_analysis(self, state: GraphState) -> dict:
        """ Starts the analysis loop by identifying the initial entity. """
        print("\n---NODE (Analysis Loop): STARTING INVESTIGATION---")
        question = state["standalone_question"]
        entities = re.findall(r"'([^']*)'|\"([^\"]*)\"", question)
        entities = [name for tpl in entities for name in tpl if name]
        if not entities:
            return {"kg_context": "Could not identify a component in your question to begin analysis.", "queried_entities": set(), "entities_to_query": []}
        print(f"---INFO: Initial analysis target: {entities[0]}---")
        return {"queried_entities": set(), "entities_to_query": [entities[0]], "kg_context": ""}

    # In your QAAnalystAgent class, replace the entire judge_completeness function with this:

    def judge_dossier_completeness(self, state: GraphState) -> dict:
        """
        The 'Critic'. Reviews the current dossier and decides if any of the components
        mentioned *within it* need a more detailed, separate dossier to be generated.
        """
        print("\n---NODE (Analysis Loop): REVIEWING DOSSIER FOR DEEP-DIVE---")
        question = state["standalone_question"]
        current_dossier = state["kg_context"]
        queried_list = list(state.get("queried_entities", set()))
    
        judge_prompt_str = """You are an expert CAE Analyst reviewing a case file (a "dossier").
        Your task is to determine if you need more detailed information about any of the components mentioned in the file to conclusively answer the user's question.

        **User's Question:**
        {question}

        **Components Already Detailed in Dossier:**
        {queried_entities_list}

        **Current Dossier:**
        {context}

        **Your Task:**
        Read the dossier. If you see a reference to a critical intermediate component (e.g., a Joint mentioned in an `[:INFLUENCES]` relationship) that is NOT already in the "Components Already Detailed" list, you should request a deep-dive on it.

        **Output Format:**
        Respond with a single JSON object with one key:
        - "next_entities": A list of NEW component names that require their own detailed dossier.
        - **If the current dossier is sufficient to answer the question, you MUST return an empty list: {{"next_entities": []}}**
        """
        
        judge_prompt = ChatPromptTemplate.from_template(judge_prompt_str)
        judge_chain = judge_prompt | self.llm.with_structured_output(schema=DossierReviewOutput)

        print("---INFO: Asking for potential deep-dive targets...---")
        response = judge_chain.invoke({
            "question": question,
            "context": current_dossier,
            "queried_entities_list": str(queried_list)
        })
        
        suggested_entities = response.next_entities if response else []
        print(f"---INFO: Judge requested deep-dive on: {suggested_entities}---")
        # We will only proceed with entities that are confirmed to exist.
        validated_entities = []
        if suggested_entities:
            for entity in suggested_entities:
                if self.neo4j_connector.entity_exists(entity):
                    validated_entities.append(entity)
                else:
                    # This is where we catch the hallucination!
                    print(f"---WARNING: Judge hallucinated a non-existent entity: '{entity}'. Discarding.---")

        print(f"---INFO: Validated entities for deep-dive: {validated_entities}---")
        # We will only proceed with entities that are confirmed to exist.
        # Return ONLY the list of valid, existing entities that are not already queried.
        validated_unique_entities = [entity for entity in validated_entities if entity not in queried_list]

        return {"entities_to_query": validated_unique_entities}

    def _query_for_more_context(self, state: GraphState) -> dict:
        """
        Performs a "deep-dive" by generating a full dossier for each entity
        requested by the judge and appending it to the main context.
        """
        print("\n---NODE (Analysis Loop): PERFORMING DEEP-DIVE---")
        entities_to_query = state["entities_to_query"]
        queried_entities = state["queried_entities"]
        current_dossier = state.get("kg_context", "")

        # Safety check: only query new entities
        new_entities = [entity for entity in entities_to_query if entity not in queried_entities]
        
        if not new_entities:
            return {"entities_to_query": []} # Your correct bug fix from before!

        print(f"---INFO: Generating enriched dossiers for: {new_entities}---")

        additional_dossiers = []
        for entity_name in new_entities:
            # Step 1: Get the standard graph dossier
            graph_dossier = self.neo4j_connector.get_dossier_for_any_entity(entity_name)
            
            # Step 2: Get the raw node data for enrichment
            raw_node_data = self.neo4j_connector.get_node_properties(entity_name)
            
            # Step 3: Call the auto-researcher to get the documentation explanation
            doc_explanation = self._get_documentation_explanation(raw_node_data)
            
            # Step 4: Combine them into a single, enriched dossier block
            enriched_dossier = graph_dossier + doc_explanation
            additional_dossiers.append(enriched_dossier)
        
        # Append the new, detailed dossiers to the main context
        updated_dossier = current_dossier + "\n\n" + "\n\n".join(additional_dossiers)
        updated_queried_set = queried_entities.union(set(new_entities))

        return {
            "kg_context": updated_dossier.strip(),
            "queried_entities": updated_queried_set
        }
    
    def _prepare_final_analysis_for_generation(self, state: GraphState) -> dict:
        """ Prepares the prompt using the combined multi-entity dossier. """
        print("\n---NODE (Holistic Analysis): PREPARING MULTI-ENTITY GENERATION---")
        context = state["kg_context"]
        question = state["question"] # Use the original question for context
        
        sys_prompt = """You are an expert Altair MotionSolve analyst. You have been provided a complete, enriched **Investigation Dossier** that includes both data from the user's model AND explanations from the official documentation.
        Synthesize ALL information in this file to provide a comprehensive, expert-level answer to the user's question. 
        Start by defining the key components using the provided explanations, 
        then explain how they are used in the model to cause the observed behavior.
    
        The question may require you to:
        - **Compare and contrast** two or more components.
        - **Explain the relationship** between different components.
        - **Trace a causal chain** that links multiple components together.

        **Crucially, pay attention to the `(Details: ...)` text on the `INFLUENCES` relationships if available.** This text explains the causal path. Use all the entries along with its specific explanation based on the Altair's help manual to construct a step-by-step explanation for your answer.

        Start by defining the key components, then trace the influence from the source (like a Motion or StateVariable) through the intermediate components (like Joints and Bodies) to the final point of query requested by user.
        """
        
        kg_prompt = ChatPromptTemplate.from_messages([
            ("system", sys_prompt),
            ("user", "Please analyze the following combined dossier to answer my question.\n\n**Combined Dossier:**\n{context}\n\n**Original Question:** {question}")
        ])
        messages = kg_prompt.invoke({"context": context, "question": question})
        return {"message": messages.to_messages()}   
    # ===================================================================
    # SECTION 5: FINAL ANSWER GENERATION AND GRAPH DEFINITION
    # ===================================================================
    
    def _generate_final_answer(self, state: GraphState) -> Generator[dict, None, None]:
        """ Generate the final answer stream using the LLM. """
        print("\n---NODE: GENERATING FINAL ANSWER---")
        message = state["message"]
        final_ans_chain = self.llm | StrOutputParser()
        final_ans = ""
        for chunk in final_ans_chain.stream(message):
            final_ans += chunk
            yield {"answer": final_ans}
        return {"answer": final_ans}
    # In your QAAnalystAgent class in agent_qa_test_analyst.py

    def _get_documentation_explanation(self, node_data: dict) -> str:
        """
        The "Auto-Researcher" tool. Takes a node's raw data, determines its type,
        fetches the relevant documentation, and uses an LLM to synthesize an
        explanation of that specific node's properties.
        """
        if not node_data:
            return ""

        # Determine the primary type of the node (e.g., "Joint", "Motion")
        node_type = next((label for label in node_data.get('_labels', []) if label != 'Node'), None)
        node_data.pop('_labels', None)  # Clean up the data dict
        if not node_type:
            return ""
        if node_type.lower() == "postrequest" and node_data.get("measurement").lower() == 'usersub':
            node_type='pr_usersub'
        term_to_lookup = node_type.lower()
        
        # The predefined map of high-frequency terms to specific URLs
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
            # Add more mappings as needed
        }

        url_to_load = url_map.get(term_to_lookup)
        if not url_to_load:
            print(f"  -> No documentation URL mapped for type '{node_type}'. Skipping enrichment.")
            return ""

        try:
            print(f"  -> Auto-researching type '{node_type}' for entity '{node_data.get('name')}'...")
            loader = WebBaseLoader([url_to_load])
            docs = loader.load()
            raw_content = "\n".join([doc.page_content for doc in docs])

            # Use a sub-LLM call to synthesize an explanation
            synthesis_prompt_template = """You are a helpful assistant. Your task is to explain a specific piece of a simulation model's data using the provided official documentation.

            **Official Documentation for a '{term}' component:**
            {documentation}

            **Data from User's Specific Component:**
            {data_context}

            **Your Task:**
            Based ONLY on the Official Documentation, provide a concise explanation of the key properties and values seen in the "Data from User's Specific Component".
            """
            
            synthesis_prompt = ChatPromptTemplate.from_template(synthesis_prompt_template)
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
    # In your QAAnalystAgent class, this is the FINAL _create_graph

    def _create_graph(self) -> StateGraph:
        workflow = StateGraph(GraphState)

        # --- Add All Nodes ---
        workflow.add_node("history_aware_retrieval", self._history_aware_retrieval)
        workflow.add_node("route_question", self.route_question)
        workflow.add_node("retrieve", self._retrieve_documents)
        workflow.add_node("prepare_rag_generation", self._prepare_rag_for_generation)
        workflow.add_node("query_kg_for_structure", self._query_kg_for_structure)
        workflow.add_node("prepare_kg_structure_generation", self._prepare_kg_structure_for_generation)
        
        # --- The Hybrid Analysis Branch Nodes ---
        workflow.add_node("run_holistic_analysis_query", self._run_holistic_analysis_query)
        workflow.add_node("judge_dossier_completeness", self.judge_dossier_completeness)
        workflow.add_node("query_for_more_context", self._query_for_more_context)
        workflow.add_node("prepare_final_analysis", self._prepare_final_analysis_for_generation)
        
        workflow.add_node("generate_answer", self._generate_final_answer)

        # --- Define the Graph's Flow ---
        workflow.set_entry_point("history_aware_retrieval")
        workflow.add_edge("history_aware_retrieval", "route_question")

        workflow.add_conditional_edges("route_question", lambda state: state["route_decision"], {
            "rag_branch": "retrieve",
            "kg_structural_branch": "query_kg_for_structure",
            "kg_analysis_branch": "run_holistic_analysis_query",
        })
        
        # RAG & Structural Branches (linear)
        workflow.add_edge("retrieve", "prepare_rag_generation")
        workflow.add_edge("prepare_rag_generation", "generate_answer")
        workflow.add_edge("query_kg_for_structure", "prepare_kg_structure_generation")
        workflow.add_edge("prepare_kg_structure_generation", "generate_answer")

        # --- The HYBRID Analysis Branch ---
        # After the initial dossier, the judge reviews it
        workflow.add_edge("run_holistic_analysis_query", "judge_dossier_completeness")
        
        # The judge's decision point (controlled by code)
        workflow.add_conditional_edges(
            "judge_dossier_completeness",
            # If the judge found new entities to deep-dive on, re-query.
            # Otherwise, the dossier is complete, so generate the answer.
            lambda state: "re-query" if state.get("entities_to_query") else "generate",
            {
                "re-query": "query_for_more_context",
                "generate": "prepare_final_analysis",
            }
        )
        
        # The deep-dive loop: after getting more context, go back to the judge for another review
        workflow.add_edge("query_for_more_context", "judge_dossier_completeness")
        
        # The exit path from the loop
        workflow.add_edge("prepare_final_analysis", "generate_answer")
        workflow.add_edge("generate_answer", END)

        return workflow.compile()
    # In your QAAnalystAgent class in agent_qa_test_analyst.py

    def _run_holistic_analysis_query(self, state: GraphState) -> dict:
        """
        Runs the initial dossier query for all entities in the question and
        initializes the state for the optional deep-dive loop.
        """
        print("\n---NODE (Holistic Analysis): GATHERING INITIAL DOSSIER---")
        question = state["standalone_question"]
        entities = re.findall(r"'([^']*)'|\"([^\"]*)\"", question)
        entities = [name for tpl in entities for name in tpl if name]

        if not entities:
            return {"kg_context": "Could not identify component names.", "queried_entities": set()}
        
        print(f"---INFO: Found {len(entities)} initial entities: {entities}---")
        
        all_dossiers = []
        for entity_name in entities:
            dossier = self.neo4j_connector.get_dossier_for_any_entity(entity_name)
            all_dossiers.append(dossier)
        
        final_context = "\n\n".join(all_dossiers)
        
        # --- CRITICAL CHANGE: Initialize the state for the loop ---
        return {
            "kg_context": final_context,
            "queried_entities": set(entities) # Prime the set with the entities we just queried
        }
    # ===================================================================
    # SECTION 6: HELPER METHODS AND MAIN EXECUTION
    # ===================================================================

    def _prepare_rag_for_generation(self, state: GraphState) -> dict:
        """
        Prepare the context and question for the LLM generation.
        Args:
            state (GraphState): The current state containing the question and documents.
        """
        context = self._format_docs_as_context(state["documents"])
        question = state["question"]
        contextualize_q_sys_prompt = """ 
        You are an expert Motion View QA Analyst. \
        The user will ask you to list relevant tests. You will have access to a set of documents about various tests. \
        Use the following context to answer the question at the end. \
        If you don't know the answer, just say that you don't know, don't try to make up an answer. \
        Be concise and to the point.
        """
        qa_prompt = ChatPromptTemplate.from_messages(
            (
                ("system", contextualize_q_sys_prompt),
                ("user", "Based on the following context, please answer the question.\n\nContext:\n{context}\n\nQuestion: {question}"),
            )
        )
        messages = qa_prompt.invoke({"context": context, "question": question})
        return {"message": messages.to_messages()}

    def _format_docs_as_context(self, docs: list[Document]) -> str:
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
    
    def _create_rag_chain(self):

        contextualize_q_sys_prompt = """ 
        You are an expert Motion View QA Analyst. \
        The user will ask you to list relevant tests. You will have access to a set of documents about various tests. \
        Use the following context to answer the question at the end. \
        If you don't know the answer, just say that you don't know, don't try to make up an answer. \
        Be concise and to the point.
        """
        qa_prompt = ChatPromptTemplate.from_messages(
            (
                ("system", contextualize_q_sys_prompt),
                ("user", "Based on the following context, please answer the question.\n\nContext:\n{context}\n\nQuestion: {question}"),
            )
        )
        

        qa_chain = (
            {"context": itemgetter("question") | self.retriever | self._format_docs_as_context, "question": itemgetter("question")}
            | qa_prompt
            | self.llm
            | StrOutputParser()
        )
        return qa_chain
    
    def process_message(self, message: str, chat_history: list):
        # Implement the logic to process the message using the vector store
        # and return a stream of response chunks.
        input = {
            "question": message,
            "chat_history": chat_history
            }
        # The stream method on a compiled graph yields the state updates from each node.
        # We need to filter for the updates from our 'generate' node to get the tokens.
        for update in self.qa_graph.stream(input):
            if "generate_answer" in update:
                yield update["generate_answer"]["answer"]
        
        # yield f"Processed message: {message}"
    def save_graph(self, filepath: Path):
        """Saves the graph structure to a file."""
        graph = self.qa_graph.get_graph()
        try:
            # Draw the graph and save it as a PNG file
            # You can also use .draw_svg() or .draw_mermaid() for other formats
            image_data = graph.draw_mermaid_png()
            
            # Save the image data to a file
            with open(filepath, "wb") as f:
                f.write(image_data)

            print(f"✅ Graph visualization saved to {filepath}")
        except Exception as e:
            print(f"❌ Could not visualize graph. Make sure you have installed graphviz.")
            print(f"   Error: {e}")
    
qa_analyst_agent = QAAnalystAgent()