# build_db.py
import os
import shutil
import argparse
import uuid
from dotenv import load_dotenv
import ollama
from chromadb import PersistentClient
import time
import nltk
from nltk.tokenize import sent_tokenize

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL") 
COLLECTION_NAME = "cat_facts_collection"
PERSIST_DIR = "chroma_db"
DOCS_PATH = "docs/cat-facts.txt"
BATCH_SIZE = 64 # Recommended batch sizes are 32-128 depending on memory and model size.

# Ensure NLTK data is available for sentence tokenization
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("Downloading NLTK 'punkt' tokenizer data...")
    nltk.download('punkt')

def read_dataset(path):
    """
    Reads the dataset from the specified path and chunks it into sentences.
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
        # Using sentence-based chunking for better context.
        # For more advanced scenarios, consider sliding window chunking with overlap.
        sentences = sent_tokenize(content)
        # Filter out empty sentences
        return [s.strip() for s in sentences if s.strip()]

def ensure_client_and_collection():
    client = PersistentClient(path=PERSIST_DIR)
    collection = client.get_or_create_collection(
    name="cat_facts_collection",
    metadata={"hnsw:space": "cosine"}
)
    return client, collection


def db_already_built():
    """
    Checks if the persistence directory exists and is non-empty.
    """
    return os.path.exists(PERSIST_DIR) and len(os.listdir(PERSIST_DIR)) > 0

def embed_batch(texts):
    """
    Calls Ollama embed API for a list of texts.
    Returns a list of embedding vectors (lists of floats).
    Includes robust error handling and unified parsing for various Ollama response formats.
    """
    try:
        resp = ollama.embed(model=EMBEDDING_MODEL, input=texts)
        # Unified parsing method for different Ollama embed response structures
        if isinstance(resp, dict):
            if "embeddings" in resp:
                return resp["embeddings"]
            elif "embedding" in resp: # For single embedding response that might be wrapped
                return [resp["embedding"]]
            elif "data" in resp and isinstance(resp["data"], list):
                return [item["embedding"] for item in resp["data"] if "embedding" in item]
        elif isinstance(resp, list) and all(isinstance(item, dict) and 'embedding' in item for item in resp):
            return [item['embedding'] for item in resp]
        elif hasattr(resp, 'embeddings') and isinstance(resp.embeddings, list): # For EmbedResponse object
            return resp.embeddings
        elif hasattr(resp, 'embedding') and isinstance(resp.embedding, list): # For single EmbedResponse object
            return [resp.embedding]
        elif hasattr(resp, 'data') and isinstance(resp.data, list): # For EmbedResponse object with data attribute
            return [item.embedding for item in resp.data]
        
        # If none of the above, log and raise error for debugging
        print(f"DEBUG: Unexpected response structure from ollama.embed. Raw response: {resp}")
        raise RuntimeError(f"Unexpected response structure from ollama.embed: {type(resp)}")
    except Exception as e:
        print(f"DEBUG: Error during embedding. Data shape: {len(texts) if isinstance(texts, list) else 'N/A'}. Raw error: {e}")
        print("DEBUG: Suggestion: Check if Ollama server is running and EMBEDDING_MODEL is correct in your .env file.")
        print("DEBUG: Alternative: Try a smaller batch size or a different embedding model.")
        raise

def main():
    parser = argparse.ArgumentParser(description="Build or rebuild a ChromaDB vector database.")
    parser.add_argument("--force", action="store_true", help="Force rebuild the database if it already exists.")
    args = parser.parse_args()

    if db_already_built():
        print(f"WARNING: Persistence directory '{PERSIST_DIR}' already exists and is non-empty.")
        if args.force:
            print("Force rebuild initiated. Deleting existing database...")
            shutil.rmtree(PERSIST_DIR)
        else:
            choice = input("Options: (S)kip, (R)ebuild, (N)ew files only (not implemented). Enter choice: ").lower()
            if choice == 's':
                print("Skipping database build.")
                return
            elif choice == 'r':
                print("Rebuilding database. Deleting existing database...")
                shutil.rmtree(PERSIST_DIR)
            else:
                print("Invalid choice or 'New files only' not implemented. Skipping database build.")
                return

    print("Reading dataset...")
    dataset = read_dataset(DOCS_PATH)
    print(f"Loaded {len(dataset)} entries from {DOCS_PATH}")

    client, collection = ensure_client_and_collection()

    print("Embedding and adding documents to Chroma in batches...")
    total = len(dataset)
    idx = 0
    while idx < total:
        batch = dataset[idx: idx + BATCH_SIZE]
        # Use UUIDs for robust ID generation to avoid collisions and ensure stability across rebuilds.
        ids = [str(uuid.uuid4()) for _ in range(len(batch))] 
        metas = [{"source": DOCS_PATH, "chunk_index": idx + j} for j in range(len(batch))]

        # embed batch
        embeddings = embed_batch(batch)
        
        if embeddings is None or len(embeddings) != len(batch):
            print(f"DEBUG: Embedding call returned unexpected result. Expected {len(batch)} embeddings, got {len(embeddings) if embeddings else 'None'}.")
            raise RuntimeError("Embedding call returned unexpected result. Check the Ollama embed API usage.")

        # Add to collection
        collection.add(
            documents=batch,
            embeddings=embeddings,
            metadatas=metas,
            ids=ids
        )

        idx += len(batch)
        print(f"Added chunk {idx}/{total} to the database")
        # Optional: Add rate control for remote Ollama servers to prevent overwhelming the server.
        # time.sleep(0.1) 

    print(f"Database persisted to directory: {PERSIST_DIR}")
    print("Build finished.")

    # Verify collection creation
    print("Contents of PERSIST_DIR:", os.listdir(PERSIST_DIR))
    client = PersistentClient(path=PERSIST_DIR)
    collections = client.list_collections()
    print(f"Collections after build: {[c.name for c in collections]}")

if __name__ == "__main__":
    start = time.time()
    main()
    print(f"Elapsed: {time.time()-start:.2f}s")
