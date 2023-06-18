from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

import pickle
import argparse
import logging

logging.basicConfig(level=logging.INFO)

def generate_vector_store(doctype=None):

    loader = DirectoryLoader('data/', glob=f"*.{doctype}",show_progress=True)
    raw_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(raw_documents)

    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)

    with open(f"{doctype}_vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dt', '--doctype', help='Type of documents to be parsed/stored.')
    args = parser.parse_args()
    
    if args:
        generate_vector_store(args.doctype)
    else:
        logging.info("No args passed exiting script")
       




