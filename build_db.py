# build_db.py
import os
from dotenv import load_dotenv
import ollama
import chromadb
from chromadb.config import Settings
import time

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")  # e.g. "llama2" or whatever you use in Ollama
COLLECTION_NAME = "cat_facts_collection"
PERSIST_DIR = "chroma_db"
DOCS_PATH = "docs/cat-facts.txt"
BATCH_SIZE = 64

def read_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        # keep each line/fact as one "chunk" (your current approach)
        return [line.strip() for line in f.readlines() if line.strip()]

def ensure_client_and_collection():
    client = chromadb.Client(Settings(persist_directory=PERSIST_DIR))
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return client, collection

def db_already_built():
    # simple check: persistent folder exists and is non-empty
    return os.path.exists(PERSIST_DIR) and len(os.listdir(PERSIST_DIR)) > 0

def embed_batch(texts):
    """
    Calls Ollama embed API for a list of texts.
    Returns a list of embedding vectors (lists of floats).
    """
    resp = ollama.embed(model=EMBEDDING_MODEL, input=texts)
    # If resp is an EmbedResponse object and has a direct 'embeddings' attribute (list of lists)
    if hasattr(resp, 'embeddings') and isinstance(resp.embeddings, list):
        return resp.embeddings
    # If resp is an EmbedResponse object and has a direct 'embedding' attribute (list of lists, less common for batch)
    elif hasattr(resp, 'embedding') and isinstance(resp.embedding, list):
        return resp.embedding
    # If resp is a list of dictionaries, extract 'embedding' from each
    elif isinstance(resp, list) and all(isinstance(item, dict) and 'embedding' in item for item in resp):
        return [item['embedding'] for item in resp]
    # Fallback for older ollama versions or different structures (e.g., single dict with 'embeddings')
    elif isinstance(resp, dict) and "embeddings" in resp:
        return resp["embeddings"]
    # If it's a single EmbedResponse object that has a 'data' attribute (list of embedding objects)
    elif hasattr(resp, 'data') and isinstance(resp.data, list):
        return [item.embedding for item in resp.data]
    raise RuntimeError(f"Unexpected response structure from ollama.embed: {type(resp)} - {resp}")

def main():
    if db_already_built():
        print(f"Persistence directory '{PERSIST_DIR}' already exists and is non-empty.")
        print("If you want to rebuild the DB, delete the folder or remove its contents, then rerun this script.")
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
        ids = [f"doc_{i}" for i in range(idx, idx + len(batch))]
        metas = [{"source": DOCS_PATH, "line": idx + j + 1} for j in range(len(batch))]

        # embed batch (may be slower if Ollama daemon needs time)
        embeddings = embed_batch(batch)
        # ensure embedding count matches
        if embeddings is None or len(embeddings) != len(batch):
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

    # Persistence is handled automatically by the client's initialization with persist_directory
    print(f"Database persisted to directory: {PERSIST_DIR}")
    print("Build finished.")

if __name__ == "__main__":
    start = time.time()
    main()
    print(f"Elapsed: {time.time()-start:.2f}s")
