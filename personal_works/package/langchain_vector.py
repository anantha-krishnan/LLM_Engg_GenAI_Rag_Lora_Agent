from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import gradio as gr

import os, glob
from dotenv import load_dotenv

import global_vars

folders = glob.glob('C:/Projects/llm_engg/llm_engineering/personal_works/')
db_name = "C:/Projects/llm_engg/llm_engineering/personal_works/vector_db"

documents = []
text_loader_kwargs={'encoding':'utf-8'}

for folder in folders:
    doc_type = os.path.basename(folder)
    loader = DirectoryLoader(folder,glob='**/*.py', loader_kwargs=text_loader_kwargs, loader_cls=TextLoader)
    folder_docs = loader.load()
    for doc in folder_docs:
        doc.metadata['doc_type']=doc_type
        documents.append(doc)    
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)


load_dotenv('')
embeddings = OpenAIEmbeddings()

if os.path.exists(db_name):
    Chroma(persist_directory=db_name,embedding_function=embeddings).delete_collection()

# Create our Chroma vectorstore
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)
llm = ChatOpenAI(model = global_vars.model_openai_4omini )
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
# the retriever is an abstraction over the VectorStore that will be used during RAG
retriever = vectorstore.as_retriever(search_kwargs={"k": 25})
# putting it together: set up the conversation chain with the GPT 4o-mini LLM, the vector store and memory
conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

def chat(question, chat_history):
    result = conversation_chain.invoke({'question':question})
    return  result["answer"]

launcher = gr.ChatInterface(chat, type='messages')
launcher.launch()
del launcher, llm, embeddings
print('exited fully')
