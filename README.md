# PDF Question Answering with RAG and Falcon-7B

This repository contains code for a PDF question-answering system using Retrieval-Augmented Generation (RAG) with the Falcon-7B Instruct Large Language Model (LLM). It leverages LangChain for document processing, embedding generation, vector storage, and LLM interaction.

## Features

* **PDF Document Loading:** Loads PDF documents from a specified directory.
* **Text Chunking:** Splits the loaded PDF text into smaller, manageable chunks for LLM processing.
* **Embeddings Generation:** Uses Hugging Face Embeddings (sentence-transformers/all-mpnet-base-v2) to create vector embeddings of the text chunks.
* **Vector Storage (FAISS):** Stores the embeddings in a FAISS vector store for efficient retrieval.
* **Falcon-7B Instruct LLM:** Utilizes the Falcon-7B Instruct model from Hugging Face Hub for question answering.
* **RetrievalQA Chain:** Combines the vector store retriever and the LLM into a question-answering chain.
* **Interactive Querying:** Provides a command-line interface for querying the PDF documents.

## Prerequisites

* Python 3.x
* Hugging Face API token
* PDF documents in a specified directory

## Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install dependencies:**

    ```bash
    pip install langchain huggingface_hub sentence-transformers faiss-cpu pypdf dotenv
    ```

4.  **Set up environment variables:**

    * Create a `.env` file in the root directory of the project.
    * Add your Hugging Face API token to the `.env` file:

        ```
        HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_token
        ```

5.  **Place PDF documents:**

    * Create a directory named `pdf` in the root of the project.
    * Place your PDF documents in the `pdf` directory.

## Usage

1.  **Run the script:**

    ```bash
    python your_script_name.py #replace your_script_name.py with the python file name.
    ```

2.  **Query the documents:**

    * The script will prompt you to enter a query.
    * Enter your question and press Enter.
    * The script will retrieve relevant information from the PDF documents and generate an answer using the Falcon-7B Instruct LLM.
    * Type 'q' and press enter to quit the interactive session.

## Code Explanation

```python
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
except:
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(vectorstore_path) # Save for later use
    print("Created and saved new vectorstore.")

# Load environment variables from the .env file
load_dotenv()
# 5. Initialize LLM (Falcon)
HUGGINGFACEHUB_API_TOKEN =  os.environ.get("HUGGINGFACEHUB_API_TOKEN")    # Replace with your Hugging Face API token
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
    result = qa_chain.run(query)
    print(result.split('Helpful Answer:')[1].strip())
