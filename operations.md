# RAG Project Operations Guide

This guide provides step-by-step instructions on how to set up, build the database for, and run the RAG (Retrieval-Augmented Generation) application.

## 1. Prerequisites

Before you begin, ensure you have the following installed:

*   **Python 3.8+**: The project is developed using Python.
*   **Ollama**: This project uses Ollama for embedding and language models.
    *   Download and install Ollama from [ollama.ai](https://ollama.ai/).
    *   Pull the necessary models. For example, to pull `nomic-embed-text` for embeddings and `llama2` for the language model:
        ```bash
        ollama pull nomic-embed-text
        ollama pull llama2
        ```
*   **NLTK Data**: The `build_db.py` script will attempt to download the `punkt` tokenizer data if it's not already present. Ensure you have an internet connection for this.

## 2. Project Setup

1.  **Clone the repository (if applicable):**
    ```bash
    git clone [repository_url]
    cd RAG_Project
    ```
2.  **Create a Python virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```
3.  **Activate the virtual environment:**
    *   **Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    *   **macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```
4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
5.  **Configure Environment Variables:**
    Create a `.env` file in the root of the project directory (`d:\RAG_Project`) and add the following, adjusting model names as per your Ollama setup:
    ```
    EMBEDDING_MODEL=nomic-embed-text
    LANGUAGE_MODEL=llama2
    ```

## 3. Building the Vector Database (`chroma_db`)

The vector database stores the embeddings of your documents, which the RAG application uses for retrieval.

*   **First-time build:**
    If the `chroma_db` directory does not exist or is empty, simply run:
    ```bash
    python build_db.py
    ```
    This will read the documents, embed them, and store them in the `chroma_db` directory.

*   **Rebuilding an existing database:**
    If the `chroma_db` directory already exists and contains data, running `python build_db.py` will prompt you with options:
    ```
    WARNING: Persistence directory 'chroma_db' already exists and is non-empty.
    Options: (S)kip, (R)ebuild, (N)ew files only (not implemented). Enter choice:
    ```
    *   Enter `s` to **Skip** the build and use the existing database.
    *   Enter `r` to **Rebuild** the database. This will delete the existing `chroma_db` directory and create a new one from scratch.
    *   The `(N)ew files only` option is currently not implemented.

*   **Force Rebuild (non-interactive):**
    To automatically delete and rebuild the database without any prompts, use the `--force` argument:
    ```bash
    python build_db.py --force
    ```

## 4. Running the RAG Application

Once the vector database is built, you can run the RAG application to ask questions based on your documents.

1.  **Ensure your Ollama server is running.**
2.  **Run the application:**
    ```bash
    python rag_app.py
    ```
3.  The application will prompt you to "Ask me a question:". Type your query and press Enter.
    ```
    Ask me a question: What are some facts about cats?
    ```
4.  The application will retrieve relevant information from the database, construct a prompt for the language model, and stream the response. It will also display the retrieved knowledge, including similarity and distance scores, for debugging purposes.

## 5. Important Notes

*   **Model Availability:** Ensure the `EMBEDDING_MODEL` and `LANGUAGE_MODEL` specified in your `.env` file are pulled and available in your Ollama instance.
*   **Debugging:** The code includes `DEBUG` print statements that provide insights into potential issues, such as unexpected Ollama response structures or embedding failures. These messages offer suggestions for troubleshooting.
*   **Hallucination Prevention:** The RAG application is designed to use *only* the provided retrieved context. If the context does not contain the answer, it will explicitly state, "I don't know based on the given context," rather than fabricating information.
*   **Chunking Strategy:** The current `build_db.py` uses sentence-based chunking. For different document types or more advanced retrieval, you might explore paragraph-based or sliding window chunking with overlap.
*   **Rate Control:** If using a remote Ollama server or experiencing rate limiting, consider adding `time.sleep()` calls in `build_db.py` during batch processing to throttle requests.
