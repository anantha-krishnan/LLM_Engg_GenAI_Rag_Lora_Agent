# ingest_MS_Tests.py
import os
import re
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

DB_DIR = (Path(__file__).parent / "chroma_db_qa_analyst").as_posix()

class VectorStore(ABC):
    @abstractmethod
    def add_documents(self, documents: List[Document], metadata: List[Dict[str, Any]], ids: List[str]) -> None:
        pass

    @abstractmethod
    def get_hybrid_search_retriever(self, query: str, alpha: float = 0.5, top_k: int = 5, where_filter=None):
        pass
    
    @abstractmethod
    def create_vector_store(self, metadata_csv_path: str, db_dir: str) -> None:
        pass

    @dataclass
    class TestMetadata:
        working_dir: str
        main_category: str
        sub_category_event: str
        model_name: str
        export_xml: str
        tags: str
        plot_reports: str
        sub_folder: str = field(default="")

class ChromaVectorStore(VectorStore):
    def __init__(self, metadata_csv_path: str, db_dir: str):
        if os.path.exists(db_dir) and os.listdir(db_dir):
            print(f"--- Loading existing vector store from {db_dir} ---")
            self.vector_store = Chroma(
                embedding_function=OpenAIEmbeddings(),
                persist_directory=db_dir
            )
        else:
            self.vector_store = self.create_vector_store(metadata_csv_path, db_dir)


    def create_vector_store(self, metadata_csv_path: str, db_dir: str) -> None:
        docs = self.create_documents(metadata_csv_path)
        print("--- Building new vector store ---")
        print(f"Creating embeddings and persisting {len(docs)} documents to {db_dir}...")
        vector_store = Chroma.from_documents(embedding=OpenAIEmbeddings(), documents=docs, persist_directory=db_dir)
        vector_store.persist()
        return vector_store

    def get_vector_store(self):
        return self.vector_store
    
    def add_documents(self, documents: List[Document], metadata: List[Dict[str, Any]], ids: List[str]) -> None:
        self.vector_store.add_documents(documents, metadata=metadata, ids=ids)
    
    def get_BM25_retriever(self, top_k: int = 5):
        all_docs = self.vector_store.get(include=["metadatas", "documents"])
        # Reconstruct the Document objects for the BM25 retriever
        bm25_documents = [Document(page_content=doc, metadata=meta) for doc, meta in zip(all_docs["documents"], all_docs["metadatas"])]
        bm25_retriever = BM25Retriever.from_documents(bm25_documents)
        bm25_retriever.k = top_k
        return bm25_retriever
    
    def get_hybrid_search_retriever(self, alpha: float = 0.5, top_k: int = 5):
        self.semantic_retriever = self.vector_store.as_retriever(search_kwargs={'k': top_k})
        self.bm25_retriever = self.get_BM25_retriever()

        self.retriever = EnsembleRetriever(
            retrievers=[self.semantic_retriever, self.bm25_retriever],
            weights=[alpha, 1 - alpha]
        )
        return self.retriever
    
    def create_documents(self, metadata_csv_path: str) -> list[Document]:
        print(f"Loading metadata from {metadata_csv_path}...")
        df = pd.read_csv(metadata_csv_path)
        df.fillna("", inplace=True)
        documents = []
        for _, row in df.iterrows():
            doc = Document(
                page_content=(
                   f"Model Name: {row['model_name']}\n"
                    f"Event Type: {row['sub_category_event']}\n"
                    f"Description and Tags: {row['tags']}"
                ),
                metadata={
                    "working_dir": row["working_dir"],
                    "main_category": row["main_category"],
                    "sub_category_event": row["sub_category_event"],
                    "model_name": row["model_name"],
                    "export_xml": row["export_xml"],
                    "sub_folder": row["sub_folder"],
                    "tags": row["tags"],
                    "plot_reports": row["plot_reports"],
                }
            )
            documents.append(doc)
        return documents

def factory_create_vector_store(metadata_csv_path: str, vector_store_type: str="chroma") -> VectorStore:
    if vector_store_type == "chroma":
        return ChromaVectorStore(metadata_csv_path, db_dir=DB_DIR)
    else:
        raise ValueError(f"Unknown vector store type: {vector_store_type}")
    
def factory_get_hybrid_retriever(VectorStoreClass:VectorStore, alpha: float=0.5, top_k: int=5):
    # check the type VectorStoreClass. whether it is ChromaVectorStore
    if isinstance(VectorStoreClass, ChromaVectorStore):
        return VectorStoreClass.get_hybrid_search_retriever(alpha=alpha, top_k=top_k)
    else:
        raise ValueError(f"Unsupported VectorStoreClass type {type(VectorStoreClass)}")
