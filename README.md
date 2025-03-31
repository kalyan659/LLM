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
    git clone -b falcon_rag https://github.com/kalyan659/LLM.git
    cd falcon_rag
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
