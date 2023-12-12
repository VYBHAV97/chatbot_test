import os
import pandas as pd
import openai
import tiktoken
import chromadb

from langchain.document_loaders import  CSVLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain


loader = CSVLoader("C:\ChatBot_Using_RGA_LLM\OhioStateReportingQuesAns.csv")
csvData = loader.load()

text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
splitData = text_splitter.split_documents(csvData);

collection_name = "erp_questions_answers_collection"
local_directory = "erp_questions_answers_vect_embedding"
persist_directory = os.path.join(os.getcwd(), local_directory)
 
openai_key="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"
embeddings = OpenAIEmbeddings(openai_api_key=openai_key, show_progress_bar=False)

vectDB = Chroma.from_documents(splitData,
                      embeddings,
                      collection_name=collection_name,
                      persist_directory=persist_directory
                      )
vectDB.persist()
vectDB = Chroma(collection_name=collection_name, persist_directory= persist_directory, embedding_function=embeddings)


memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
chatQA = ConversationalRetrievalChain.from_llm(
            OpenAI(openai_api_key=openai_key,
               temperature=0.7, model_name="gpt-4"), 
            vectDB.as_retriever(), 
            memory=memory)

chat_history = []
qry = ""
while qry != 'done':
    qry = input('Question: ')
    if qry != exit:
        response = chatQA({"question": qry, "chat_history": chat_history})
        print("Answer: " + response["answer"])