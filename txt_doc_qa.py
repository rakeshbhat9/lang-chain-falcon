
import os
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
import pickle

from langchain import HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]

# --------------------------------------------------------------------------
def get_chain(vectorstore):
    llm = HuggingFaceHub(
    repo_id="tiiuae/falcon-7b", model_kwargs={"temperature": 0.1, "max_new_tokens":500},
    huggingfacehub_api_token = HUGGINGFACEHUB_API_TOKEN
)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":2})

    qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever)

    return qa_chain

# --------------------------------------------------------------------------

if __name__ == "__main__":


    with open("txt_vectorstore.pkl", "rb") as f:
        vectorstore = pickle.load(f)

    qa_chain = get_chain(vectorstore)
    chat_history = []

    print("Welcome to the chatbot! Type 'exit' to exit the chatbot.")
    
    while True:
        print(f"{str(datetime.now())} Human:")
        question = input()
        if question=="exit":
            break
        result = qa_chain({"question": question, "chat_history": chat_history})
        chat_history.append((question, result["answer"]))
        print(f"{str(datetime.now())} AI:")
        print(result["answer"])
