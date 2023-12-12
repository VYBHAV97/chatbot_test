from flask import Flask, request, jsonify
from flask_cors import CORS                 
import os
import pandas as pd
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain

app = Flask(__name__)
CORS(app)  

# loader = CSVLoader("C:\ChatBot_Using_RGA_LLM\OhioStateReportingQuesAns.csv")
# csvData = loader.load()

# text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
# splitData = text_splitter.split_documents(csvData)

# collection_name = "erp_questions_answers_collection"
# local_directory = "erp_questions_answers_vect_embedding"
# persist_directory = os.path.join(os.getcwd(), local_directory)

# openai_key = "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"
# embeddings = OpenAIEmbeddings(openai_api_key=openai_key, show_progress_bar=False)

# vectDB = Chroma.from_documents(splitData,
#                                embeddings,
#                                collection_name=collection_name,
#                                persist_directory=persist_directory
#                                )
# vectDB.persist()
# vectDB = Chroma(collection_name=collection_name, persist_directory=persist_directory, embedding_function=embeddings)

# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# chatQA = ConversationalRetrievalChain.from_llm(
#     OpenAI(openai_api_key=openai_key,
#            temperature=0.0, model_name="gpt-4"),
#     vectDB.as_retriever(),
#     memory=memory
# )

@app.route('/')
def index():
    return "ChatBot Flask Server is running!"

# @app.route('/ask', methods=['POST'])
# def ask_question():
#     data = request.get_json()
#     question = data.get('question', '')
#     chat_history = data.get('chat_history', [])

#     if question:
#         response = chatQA({"question": question, "chat_history": chat_history})
#         answer = response["answer"]
#         chat_history.append({"question": question, "answer": answer})
#         return jsonify({"answer": answer, "chat_history": chat_history})

#     return jsonify({"error": "Invalid input"})


if __name__ == '__main__':
    app.run(debug=True)
