# agent_qa_test_analyst.py
from pathlib import Path

import global_vars
from ingest_ms_tests import factory_create_vector_store, factory_get_hybrid_retriever

from operator import itemgetter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.docstore.document import Document

# --- AGENT SETUP ---
METADATA_CSV = (Path(__file__).parent / "MS_Tests_Metadata.csv").as_posix()
LLM_MODEL_NAME = global_vars.model_openai_4omini

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
        self.qa_chain = self._create_rag_chain()

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
        def format_docs_as_context(docs: list[Document]) -> str:
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

        qa_chain = (
            {"context": itemgetter("question") | self.retriever | format_docs_as_context, "question": itemgetter("question")}
            | qa_prompt
            | self.llm
            | StrOutputParser()
        )
        return qa_chain
    
    def process_message(self, message: str, chat_history: list):
        # Implement the logic to process the message using the vector store
        # and return a stream of response chunks.
        # This is a placeholder implementation.
        return self.qa_chain.stream({"question": message})
        # yield f"Processed message: {message}"

qa_analyst_agent = QAAnalystAgent()