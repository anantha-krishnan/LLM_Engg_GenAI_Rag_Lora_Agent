from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

import os, glob
from dotenv import load_dotenv

import global_vars

folders = glob.glob('C:/Projects/llm_engg/llm_engineering/week5/knowledge-base/*')

documents = []
text_loader_kwargs={'encoding':'utf-8'}

for folder in folders:
    doc_type = os.path.basename(folder)
    loader = DirectoryLoader(folder,glob='*.md', loader_kwargs=text_loader_kwargs, loader_cls=TextLoader)
    folder_docs = loader.load()
    for doc in folder_docs:
        doc.metadata['doc_type']=doc_type
        documents.append(doc)    
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)


load_dotenv('')
embeddings = OpenAIEmbeddings()
a=0
