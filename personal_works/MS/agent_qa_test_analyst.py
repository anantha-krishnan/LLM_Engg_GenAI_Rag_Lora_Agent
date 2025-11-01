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

from langgraph.graph import StateGraph, END


# --- AGENT SETUP ---
METADATA_CSV = (Path(__file__).parent / "MS_Tests_Metadata.csv").as_posix()
LLM_MODEL_NAME = global_vars.model_openai_4omini

class GraphState(TypedDict):
    # Original fields
    route_decision: str
    question: str
    retrieved_context: str
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
        question = state["question"].lower()
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
        contextualize_q_system_prompt = """Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."""
        contextualize_q_prompt = ChatPromptTemplate.from_messages([("system", contextualize_q_system_prompt), MessagesPlaceholder(variable_name="chat_history"), ("human", "{question}")])
        history_retriever = contextualize_q_prompt | self.llm | StrOutputParser()
        retrieved_context = history_retriever.invoke({"chat_history": chat_history, "question": question})
        return {"retrieved_context": retrieved_context or question}

    def _retrieve_documents(self, state: GraphState) -> dict:
        """ Retrieve relevant documents from vector store. """
        print("\n---NODE (RAG): RETRIEVING DOCUMENTS---")
        question = state["retrieved_context"]
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
        """ [THIS WAS THE MISSING FUNCTION] Queries Neo4j for structural context. """
        print("\n---NODE (KG Structure): QUERYING NEO4J---")
        question = state["question"]
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
        question = state["question"]
        entities = re.findall(r"'([^']*)'|\"([^\"]*)\"", question)
        entities = [name for tpl in entities for name in tpl if name]
        if not entities:
            return {"kg_context": "Could not identify a component in your question to begin analysis.", "queried_entities": set(), "entities_to_query": []}
        print(f"---INFO: Initial analysis target: {entities[0]}---")
        return {"queried_entities": set(), "entities_to_query": [entities[0]], "kg_context": ""}

    # In your QAAnalystAgent class, replace the entire judge_completeness function with this:

    def judge_completeness(self, state: GraphState) -> dict:
        """
        'Critic' node. Its only job is to identify NEW entities to query.
        The decision to stop or continue is made by the graph's edge logic.
        """
        print("\n---NODE (Analysis Loop): IDENTIFYING NEXT STEPS---")
        question = state["question"]
        current_context = state["kg_context"]
        queried_list = list(state.get("queried_entities", set()))

        judge_prompt_str = """You are a methodical Root Cause Analyst for CAE simulations.
        Your task is to identify if there are any NEW, relevant components to investigate to answer the user's question.

        **User's Question:**
        {question}

        **Components Already Investigated:**
        {queried_entities_list}

        **Current Dossier of Information:**
        {context}

        **Your Task:**
        Based on the dossier, identify the names of any components that are relevant to the question but are NOT in the "Components Already Investigated" list.
        
        **Output Format:**
        Respond with a single JSON object containing one key:
        - "next_entities": A list of NEW component names (strings) to query for more context.
        - **If you cannot find any new, relevant, un-investigated components, you MUST return an empty list: {{"next_entities": []}}**
        """
        
        judge_prompt = ChatPromptTemplate.from_template(judge_prompt_str)
        # Ensure the model is likely to return valid JSON
        judge_chain = judge_prompt | self.llm.with_structured_output(
            schema={"next_entities": "list[str]"},
            method="json_mode"
        )

        print("---INFO: Asking for next entities to investigate...---")
        response = judge_chain.invoke({
            "question": question,
            "context": current_context,
            "queried_entities_list": str(queried_list)
        })
        
        entities = response.get("next_entities", [])
        print(f"---INFO: Judge identified next entities: {entities}---")

        # The function now only returns the list of entities to query.
        return {"entities_to_query": entities}
    
    def _query_kg_for_more_context(self, state: GraphState) -> dict:
        """ 'Action' node that queries the KG for entities requested by the judge. """
        print("\n---NODE (Analysis Loop): GATHERING MORE EVIDENCE---")
        entities_to_query = state["entities_to_query"]
        queried_entities = state["queried_entities"]
        current_context = state.get("kg_context", "")
        new_entities = [entity for entity in entities_to_query if entity not in queried_entities]
        if not new_entities:
            print("---INFO: No new entities to query. Halting loop.---")
            return {"judge_decision": "generate", "queried_entities": queried_entities}

        print(f"---INFO: Fetching new context for: {new_entities}---")
        newly_found_context_parts = []
        for entity_name in new_entities:
            cypher_query = """MATCH (n {name: $name}) OPTIONAL MATCH (n)-[r]-(neighbor) WITH n, r, neighbor OPTIONAL MATCH (n)-[:HAS_COMPONENT]->(oc:OutputComponent) RETURN n, r, neighbor, collect(oc) as components"""
            results = self.neo4j_connector.query(cypher_query, parameters={"name": entity_name})
            structural_results, data_results = [], []
            for record in results:
                structural_results.append({k: v for k, v in record.items() if k != 'components'})
                for comp_node in record['components']: data_results.append({'n': comp_node})
            newly_found_context_parts.append(self.neo4j_connector.format_results_to_text(structural_results))
            newly_found_context_parts.append(self.neo4j_connector.format_results_to_text(data_results))
        
        updated_context = current_context + "\n\n--- Additional Context ---\n" + "\n".join(filter(None, newly_found_context_parts))
        updated_queried_set = queried_entities.union(set(new_entities))
        return {"kg_context": updated_context.strip(), "queried_entities": updated_queried_set}

    
    def _prepare_final_analysis_for_generation(self, state: GraphState) -> dict:
        """ Prepares the final, complete dossier for the answer synthesizer. """
        print("\n---NODE (Analysis Loop): PREPARING FINAL DOSSIER---")
        context = state["kg_context"]
        question = state["question"]
        sys_prompt = """You are an expert Altair MotionSolve simulation analyst performing a final root cause analysis.
        You have a complete dossier of information gathered from a knowledge graph.
        Synthesize all information into a single, comprehensive, and conclusive answer.
        1. Summarize the Symptom (from the numerical data).
        2. Present the Evidence (from the structural context).
        3. State your Conclusion (the causal link).
        4. Provide Recommendations.
        """
        kg_prompt = ChatPromptTemplate.from_messages([("system", sys_prompt), ("user", "**Dossier:**\n{context}\n\n**Original Question:** {question}")])
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

    def _create_graph(self) -> StateGraph:
        """
        Compiles all the nodes and edges into the final, runnable graph.
        This workflow features three distinct branches: RAG, Structural KG, and Causal Analysis KG.
        """
        workflow = StateGraph(GraphState)

        # --- 1. Add All Nodes to the Graph ---

        # Main router
        workflow.add_node("route_question", self.route_question)

        # Nodes for the RAG branch
        workflow.add_node("history_aware_retrieval", self._history_aware_retrieval)
        workflow.add_node("retrieve", self._retrieve_documents)
        workflow.add_node("prepare_rag_generation", self._prepare_rag_for_generation)

        # Nodes for the simple Structural KG branch
        workflow.add_node("query_kg_for_structure", self._query_kg_for_structure)
        workflow.add_node("prepare_kg_structure_generation", self._prepare_kg_structure_for_generation)
        
        # Nodes for the advanced Causal Analysis KG branch (the "Dossier" method)
        workflow.add_node("run_holistic_analysis_query", self._run_holistic_analysis_query)
        workflow.add_node("prepare_final_analysis", self._prepare_final_analysis_for_generation)
        
        # Final shared node for generating the answer
        workflow.add_node("generate_answer", self._generate_final_answer)

        # --- 2. Define the Graph's Flow (Edges) ---

        # The graph starts at the router
        workflow.set_entry_point("route_question")

        # The router directs the flow to one of the three branches
        workflow.add_conditional_edges(
            "route_question",
            lambda state: state["route_decision"],
            {
                "rag_branch": "history_aware_retrieval",
                "kg_structural_branch": "query_kg_for_structure",
                "kg_analysis_branch": "run_holistic_analysis_query",
            }
        )
        
        # Define the linear flow for the RAG branch
        workflow.add_edge("history_aware_retrieval", "retrieve")
        workflow.add_edge("retrieve", "prepare_rag_generation")
        workflow.add_edge("prepare_rag_generation", "generate_answer")

        # Define the linear flow for the Structural KG branch
        workflow.add_edge("query_kg_for_structure", "prepare_kg_structure_generation")
        workflow.add_edge("prepare_kg_structure_generation", "generate_answer")

        # Define the linear flow for the Causal Analysis KG branch
        workflow.add_edge("run_holistic_analysis_query", "prepare_final_analysis")
        workflow.add_edge("prepare_final_analysis", "generate_answer")
        
        # The final node marks the end of the process
        workflow.add_edge("generate_answer", END)

        return workflow.compile()

    def _run_holistic_analysis_query(self, state: GraphState) -> dict:
        """
        Runs the new, MARKER-AWARE multi-hop query to gather the complete dossier
        for the PostRequest mentioned in the user's question.
        """
        print("\n---NODE (Holistic Analysis): GATHERING MARKER-AWARE DOSSIER---")
        question = state["question"]
        
        # Extract the PostRequest name, e.g., 'Body 1-left(Output 0)'
        entities = re.findall(r"'([^']*)'|\"([^\"]*)\"", question)
        if not entities:
            return {"kg_context": "Could not identify a component name in the question to analyze."}
        
        request_name = entities[0][0]
        # request_name = [name for tpl in request_name for name in tpl if name]

        print(f"---INFO: Running holistic analysis for PostRequest: {request_name}---")
        
        # Call our new, superior connector method
        dossier = self.neo4j_connector.get_full_context_for_output(request_name)
        
        print(f"---INFO: Generated Dossier:\n{dossier}---")
        return {"kg_context": dossier}   
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