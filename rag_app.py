# rag_app.py
import os
from dotenv import load_dotenv
import ollama
import chromadb
from chromadb.config import Settings

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
LANGUAGE_MODEL = os.getenv("LANGUAGE_MODEL")
COLLECTION_NAME = "cat_facts_collection"
PERSIST_DIR = "chroma_db"
TOP_K = 3

def ensure_client_and_collection():
    # Connect to persistent Chroma on disk
    client = chromadb.Client(Settings(persist_directory=PERSIST_DIR))
    # get_collection will raise if not present; use get_or_create if you prefer fallback
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception:
        # if not found, create empty collection (but ideally you should run build_db.py first)
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return client, collection

def embed_query(query):
    resp = ollama.embed(model=EMBEDDING_MODEL, input=query)
    # extract embedding similarly to build script
    if isinstance(resp, dict) and "embeddings" in resp:
        return resp["embeddings"][0]
    return resp[0] if isinstance(resp, (list, tuple)) else resp

def retrieve(collection, query, top_n=TOP_K):
    q_emb = embed_query(query)
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=top_n,
    )
    # results likely has keys 'documents' and 'distances' (list of lists)
    retrieved = []
    docs = results.get("documents", [[]])
    dists = results.get("distances", [[]])
    if docs and len(docs) > 0:
        for i in range(len(docs[0])):
            chunk = docs[0][i]
            distance = dists[0][i] if dists and len(dists) > 0 and len(dists[0]) > i else None
            similarity = None
            if distance is not None:
                # convert distance to a positive similarity-like score
                similarity = 1.0 / (1.0 + distance)
            retrieved.append((chunk, similarity))
    return retrieved

def build_instruction_prompt(retrieved_knowledge):
    # Build a system instruction that contains only the retrieved contexts
    context_lines = "\n".join([f" - {chunk}" for chunk, _ in retrieved_knowledge])
    prompt = f"""You are a helpful chatbot.
Use only the following pieces of context to answer the question. Don't make up any new information:
{context_lines}
"""
    return prompt

def chat_with_model(system_prompt, user_query):
    stream = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
        stream=True,
    )
    # stream yields chunks; print in realtime
    print("Chatbot response:")
    text = ""
    for chunk in stream:
        # chunk structure follows your earlier usage: chunk['message']['content']
        msg = chunk.get("message", {}).get("content", "")
        print(msg, end="", flush=True)
        text += msg
    print()  # newline
    return text

def main():
    if not os.path.exists(PERSIST_DIR) or len(os.listdir(PERSIST_DIR)) == 0:
        print(f"Persistence directory '{PERSIST_DIR}' not found or empty.")
        print("Run build_db.py first to create and persist the collection.")
        return

    client, collection = ensure_client_and_collection()
    query = input("Ask me a question: ").strip()
    if not query:
        print("Empty query, exiting.")
        return

    retrieved = retrieve(collection, query, top_n=TOP_K)
    if not retrieved:
        print("No relevant documents found.")
        return

    print("Retrieved knowledge:")
    for chunk, sim in retrieved:
        sim_str = f"{sim:.2f}" if sim is not None else "N/A"
        print(f" - (similarity: {sim_str}) {chunk}")

    system_prompt = build_instruction_prompt(retrieved)
    # Ask the LLM using the retrieved context
    chat_with_model(system_prompt, query)

if __name__ == "__main__":
    main()
