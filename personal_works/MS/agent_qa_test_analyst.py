# agent_qa_test_analyst.py
from pathlib import Path
from typing import List, TypedDict, Generator, Optional
from operator import itemgetter


import global_vars
from ingest_ms_tests import factory_create_vector_store, factory_get_hybrid_retriever

from langchain_core.messages import BaseMessage
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.docstore.document import Document
from langchain.chains import create_history_aware_retriever

from langgraph.graph import StateGraph, END


# --- AGENT SETUP ---
METADATA_CSV = (Path(__file__).parent / "MS_Tests_Metadata.csv").as_posix()
LLM_MODEL_NAME = global_vars.model_openai_4omini

class GraphState(TypedDict):
    question: str
    retrieved_context: str
    chat_history: Optional[List[BaseMessage]]
    documents: List[Document]
    message: List[BaseMessage]
    answer: str

class QAAnalystAgent:
    def __init__(self):
        vs = factory_create_vector_store(
            metadata_csv_path=METADATA_CSV,
            vector_store_type="chroma"
        )
        # Get the underlying retriever
        self.retriever = factory_get_hybrid_retriever(vs, alpha=0.5, top_k=500)
        self.llm = ChatOpenAI(
            model_name=LLM_MODEL_NAME,
            temperature=0.3,
            streaming=True,
        )
        self.qa_graph = self._create_graph()

    def _history_aware_retrieval(self, state: GraphState) -> dict:
        """
        Retrieve relevant documents based on the question and chat history in the state.
        Args:
            state (GraphState): The current state containing the question and chat history.
        Returns:
            dict: A dictionary containing the retrieved documents.
        """
        
        print("\n---NODE: HISTORY-AWARE RETRIEVAL---")
        question = state["question"]
        chat_history = state.get("chat_history", [])
        
        contextualize_q_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )
        history_retriever = contextualize_q_prompt | self.llm | StrOutputParser()

        retrieved_context = history_retriever.invoke({"chat_history": chat_history, "question": question})

        if not retrieved_context or not retrieved_context.strip():
            print("---WARNING: Rephrasing returned an empty string. Falling back to original question.---")
            final_question = question
        else:
            final_question = retrieved_context

        
        #print(f"---INFO: Question for retrieval: {final_question}---")
        return {"retrieved_context": final_question}

        
    def _retrieve_documents(self, state: GraphState) -> dict:
        """
        Retrieve relevant documents based on the question in the state.
        Args:
            state (GraphState): The current state containing the question.
        Returns:
            dict: A dictionary containing the retrieved documents.
        """
        print("\n---NODE: RETRIEVING DOCUMENTS---")
        question = state["retrieved_context"]
        docs = self.retriever.invoke(question)
        return {"documents": docs}
    
    def _generate_final_answer(self, state: GraphState) -> Generator[dict, None, None]:
        """
        Generate the final answer using the LLM based on the prepared message.
        Args:
            state (GraphState): The current state containing the message.
        Returns:
            dict: A dictionary containing the final answer.
        """
        print("\n---NODE: GENERATING FINAL ANSWER---")
        message = state["message"]
        final_ans_chain = self.llm | StrOutputParser()
        final_ans = ""
        
        for chunk in final_ans_chain.stream(message):
            final_ans += chunk
            yield {"answer": final_ans}
        
        return {"answer": final_ans}
        
    def _create_graph(self) -> StateGraph:
        workflow = StateGraph(GraphState)

        workflow.add_node("history_aware_retrieval", self._history_aware_retrieval)
        workflow.add_node("retrieve", self._retrieve_documents)
        workflow.add_node("prepare_generation", self._prepare_for_generation)
        workflow.add_node("generate_answer", self._generate_final_answer)

        workflow.set_entry_point("history_aware_retrieval")
        workflow.add_edge("history_aware_retrieval", "retrieve")
        workflow.add_edge("retrieve", "prepare_generation")
        workflow.add_edge("prepare_generation", "generate_answer")
        workflow.add_edge("generate_answer", END)

        return workflow.compile()

    def _prepare_for_generation(self, state: GraphState) -> dict:
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
