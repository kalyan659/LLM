# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 12:45:12 2025

@author: kalyanbrat
"""

import os
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS  # Or Chroma, Pinecone, etc.
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain.document_loaders import PyPDFLoader  # For loading PDFs
from langchain.text_splitter import CharacterTextSplitter # For splitting text
import warnings
warnings.filterwarnings("ignore")

# 1. Load PDF Documents
pdf_directory = "pdf"  # Replace with the actual path
loaders = [PyPDFLoader(os.path.join(pdf_directory, fn)) for fn in os.listdir(pdf_directory) if fn.endswith(".pdf")]
documents = []
for loader in loaders:
    documents.extend(loader.load())

# 2. Split Text into Chunks (Important for LLMs)
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)  # Adjust chunk size as needed
docs = text_splitter.split_documents(documents)

# 3. Initialize Embeddings Model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"  # Or another suitable model
)

# 4. Create or Load Vectorstore (FAISS Example)
vectorstore_path = "faiss_index" # Path to save/load FAISS index
try:
    vectorstore = FAISS.load_local(vectorstore_path, embeddings)
    print("Successfully loaded existing vectorstore.")
    #print(0/0)
except:
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(vectorstore_path) # Save for later use
    print("Created and saved new vectorstore.")

# Load environment variables from the .env file
load_dotenv()
# 5. Initialize LLM (Falcon)
HUGGINGFACEHUB_API_TOKEN =  os.environ.get("HUGGINGFACEHUB_API_TOKEN")   # Replace with your Hugging Face API token
llm = HuggingFaceHub(
    repo_id="tiiuae/falcon-7b-instruct",  # Or another Falcon model
    model_kwargs={"temperature": 0.2, "max_new_tokens": 512},
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
)

# 6. Create RetrievalQA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
)

# 7. Query the Chain
while 1:
    # Get user input
    query = input("Please enter something: ")
    if query == 'q':
        break
    #query = "What is the main topic of this document?"  # Example query. Replace as needed.
    result = qa_chain.run(query)
    print(result.split('Helpful Answer:')[1].strip())
    
