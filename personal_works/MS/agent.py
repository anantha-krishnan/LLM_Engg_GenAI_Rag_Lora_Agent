# agent.py
import os
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.chains import create_history_aware_retriever
from langchain.docstore.document import Document

from langchain.retrievers import BM25Retriever, EnsembleRetriever
# ----------------------------------------------------
import global_vars

# --- AGENT SETUP ---
DB_DIR = "./chroma_db_mdl"
LLM_MODEL = global_vars.model_openai_4omini

class MdlAgent:
    def __init__(self):
        if not os.path.exists(DB_DIR):
            raise FileNotFoundError(
                f"Vector store not found at {DB_DIR}. "
                "Please run `python ingest.py` first."
            )
        
        self.vector_store = Chroma(
            persist_directory=DB_DIR, 
            embedding_function=OpenAIEmbeddings()
        )
        self.llm = ChatOpenAI(model=LLM_MODEL, temperature=0.3, streaming=True)
        knn_k = 50  # Number of nearest neighbors to retrieve
        # --- START OF BM25 & ENSEMBLE RETRIEVER SETUP ---

        # 1. Initialize the semantic retriever (what we had before)
        semantic_retriever = self.vector_store.as_retriever(search_kwargs={'k':knn_k})

        # 2. Initialize the BM25 keyword retriever
        # To do this, we need all the documents. We can get them from our vector store.
        print("Initializing BM25 Retriever...")
        all_docs_from_db = self.vector_store.get(include=["metadatas", "documents"])
        
        # Reconstruct the Document objects for the BM25 retriever
        reconstructed_docs = []
        for i, doc_text in enumerate(all_docs_from_db['documents']):
            reconstructed_docs.append(
                Document(
                    page_content=doc_text, 
                    metadata=all_docs_from_db['metadatas'][i]
                )
            )
        
        bm25_retriever = BM25Retriever.from_documents(reconstructed_docs)
        bm25_retriever.k = knn_k # Match the k value for consistency

        # 3. Initialize the EnsembleRetriever
        # This retriever combines the results of the other two
        # The weights determine how much influence each retriever has. 0.5/0.5 is a good start.
        self.retriever = EnsembleRetriever(
            retrievers=[semantic_retriever, bm25_retriever], weights=[0.5, 0.5]
        )
        print("Hybrid search (EnsembleRetriever) is ready.")

        # --- END OF NEW SETUP ---
        
        self.chain = self._create_full_rag_chain()

    def _format_docs(self, docs):
        """Formats docs for the prompt. No changes needed here."""
        if not docs:
            return "No relevant components were found in the library."
        
        formatted_docs = []
        for i, doc in enumerate(docs):
            meta = doc.metadata
            description_text = "N/A"
            if 'Description: ' in doc.page_content:
                description_text = doc.page_content.split('Description: ')[-1]

            doc_str = (
                f"--- Component {i+1} ---\n"
                f"Type: {meta.get('model_type', 'N/A')}\n"
                f"Side: {meta.get('side', 'N/A')}\n"
                f"Category: {meta.get('category1', '')} {meta.get('category2', '')}\n"
                f"File Path: {meta.get('file_path', 'N/A')}\n"
                f"Description: {description_text}"
            )
            formatted_docs.append(doc_str)

        return "\n\n".join(formatted_docs)

    def _create_full_rag_chain(self):
        """Creates the full RAG chain. No changes needed here, it just uses self.retriever."""
        
        contextualize_q_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )
        
        history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, contextualize_q_prompt
        )

        qa_system_prompt = """You are an expert assistant for Altair MotionView. Your goal is to help users find vehicle dynamic components.
        You must answer questions based ONLY on the context provided.
        - The user may refer to a component by its number in the list. Use the history to understand which component they mean.
        - If the user asks for options, list them clearly using the information in the context.
        - If the context is empty, state that you could not find any matching components."""
        
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "Question: {input}\n\n--- CONTEXT ---\n{context}"),
            ]
        )

        def format_docs_in_chain(chain_input):
            docs = chain_input['context']
            formatted_context = self._format_docs(docs)
            return {
                "input": chain_input['input'],
                "chat_history": chain_input['chat_history'],
                "context": formatted_context
            }

        final_chain = (
            RunnablePassthrough.assign(context=history_aware_retriever)
            | format_docs_in_chain
            | qa_prompt
            | self.llm
            | StrOutputParser()
        )

        return final_chain

    def invoke(self, question: str, chat_history: list):
        """Invokes the RAG chain with the user's question and history."""
        return self.chain.stream({
            "input": question,
            "chat_history": chat_history
        })

# Create a single instance of the agent to be used by the UI
mdl_agent = MdlAgent()