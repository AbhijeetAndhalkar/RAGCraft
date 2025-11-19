# /mnt/data/rag_app.py
import os
from dotenv import load_dotenv
import ollama
from chromadb import PersistentClient

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
LANGUAGE_MODEL = os.getenv("LANGUAGE_MODEL")
COLLECTION_NAME = "cat_facts_collection"
PERSIST_DIR = "chroma_db"
TOP_K = 3

def ensure_client_and_collection():
    """
    Ensures a ChromaDB client and collection are initialized and configured to use cosine distance.
    """
    client = PersistentClient(path=PERSIST_DIR)
    collections = client.list_collections()
    print(f"DEBUG: Collections visible before get_or_create_collection: {[c.name for c in collections]}")

    # Request collection with cosine metric for HNSW index.
    # The metadata key "hnsw:space" tells the HNSW index to use cosine distance.
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    return client, collection

def embed_query(query):
    """
    Calls Ollama embed API for a single query.
    Returns a single embedding vector (list of floats).
    """
    try:
        resp = ollama.embed(model=EMBEDDING_MODEL, input=query)
        # Unified parsing for different response shapes
        if isinstance(resp, dict):
            if "embedding" in resp:
                return resp["embedding"]
            elif "embeddings" in resp and isinstance(resp["embeddings"], list) and len(resp["embeddings"]) > 0:
                return resp["embeddings"][0]
            elif "data" in resp and isinstance(resp["data"], list) and len(resp["data"]) > 0:
                return resp["data"][0]["embedding"] if "embedding" in resp["data"][0] else None
        elif hasattr(resp, 'embedding') and isinstance(resp.embedding, list):
            return resp.embedding
        elif hasattr(resp, 'embeddings') and isinstance(resp.embeddings, list) and len(resp.embeddings) > 0:
            return resp.embeddings[0]
        elif hasattr(resp, 'data') and isinstance(resp.data, list) and len(resp.data) > 0:
            return resp.data[0].embedding

        print(f"DEBUG: Unexpected response structure from ollama.embed for query. Raw response: {resp}")
        raise RuntimeError(f"Unexpected response structure from ollama.embed for query: {type(resp)}")
    except Exception as e:
        print(f"DEBUG: Error during query embedding. Raw error: {e}")
        print("DEBUG: Suggestion: Check if Ollama server is running and EMBEDDING_MODEL is correct in your .env file.")
        raise

def retrieve(collection, query, top_n=TOP_K):
    """
    Retrieves top_n relevant documents from the ChromaDB collection based on the query.
    Returns a list of dictionaries, each containing chunk, distance, similarity, and metadata.

    IMPORTANT: This function expects the collection/index to use COSINE distance (hnsw:space='cosine').
    When using that metric, Chroma typically returns distances as:
        cosine_distance = 1 - cosine_similarity
    so we convert back with cosine_similarity = 1 - distance.
    """
    q_emb = embed_query(query)
    if q_emb is None:
        print("DEBUG: Query embedding failed, cannot retrieve documents.")
        return []

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=top_n,
        include=['documents', 'distances', 'metadatas']
    )
    
    retrieved = []
    docs = results.get("documents", [[]])
    dists = results.get("distances", [[]])
    metas = results.get("metadatas", [[]])

    if docs and len(docs) > 0:
        for i in range(len(docs[0])):
            chunk = docs[0][i]
            distance = dists[0][i] if dists and len(dists) > 0 and len(dists[0]) > i else None
            metadata = metas[0][i] if metas and len(metas) > 0 and len(metas[0]) > i else {}
            
            similarity = None
            if distance is not None:
                # Convert cosine distance (returned by Chroma with hnsw:space='cosine')
                # back into cosine similarity.
                # cosine_distance = 1 - cosine_similarity  =>  cosine_similarity = 1 - distance
                cosine_similarity = 1.0 - distance

                # Clamp to valid range just in case:
                cosine_similarity = max(-1.0, min(1.0, cosine_similarity))

                # Many embeddings typically produce cosines in [0,1]; if you want strictly 0..1:
                # similarity_0_1 = (cosine_similarity + 1.0) / 2.0
                similarity = cosine_similarity

            retrieved.append({"chunk": chunk, "distance": distance, "similarity": similarity, "metadata": metadata})
    return retrieved

def build_instruction_prompt(retrieved_knowledge):
    """
    Builds a system instruction prompt for the LLM, incorporating only the retrieved context.
    Enforces hallucination prevention rules.
    """
    context_lines = "\n".join([f" - {item['chunk']}" for item in retrieved_knowledge])
    prompt = f"""You are a helpful chatbot.
Use ONLY the following pieces of context to answer the question.
If the context does not contain the answer, reply with: "I don't know based on the given context."
Do NOT fabricate facts. You may summarize, combine pieces, or explain—but not invent.

Retrieved Context:
{context_lines}
"""
    return prompt

def chat_with_model(system_prompt, user_query):
    """
    Interacts with the Ollama language model using the provided system prompt and user query.
    Streams the response and includes error handling.
    """
    try:
        stream = ollama.chat(
            model=LANGUAGE_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ],
            stream=True,
        )
        print("Chatbot response:")
        text = ""
        for chunk in stream:
            msg = chunk.get("message", {}).get("content", "")
            print(msg, end="", flush=True)
            text += msg
        print()  # newline
        return text
    except Exception as e:
        print(f"DEBUG: Error during chat with Ollama model. Raw error: {e}")
        print("DEBUG: Suggestion: Check if Ollama server is running and LANGUAGE_MODEL is correct in your .env file.")
        print("DEBUG: Alternative: Try a different language model or simplify the query.")
        raise

def main():
    """
    Main function to run the RAG application.
    Initializes ChromaDB, retrieves documents, builds a prompt, and chats with the LLM.
    """
    if not os.path.exists(PERSIST_DIR):
        print(f"Persistence directory '{PERSIST_DIR}' not found.")
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
        print("I don't know based on the given context.") # Hallucination prevention if no docs found
        return

    # Clean terminal display — no similarity, no distances, no explanations
    print("\nUsing retrieved knowledge...\n")
    for item in retrieved:
        print(f" - {item['chunk']}")

    print()  # Empty line for spacing


    system_prompt = build_instruction_prompt(retrieved)
    chat_with_model(system_prompt, query)

if __name__ == "__main__":
    main()
