import os
import sys
from dotenv import load_dotenv
import ollama
from chromadb import PersistentClient

load_dotenv()

# Variables
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
LANGUAGE_MODEL = os.getenv("LANGUAGE_MODEL", "llama2")
COLLECTION_NAME = "cat_facts_collection"
PERSIST_DIR = "chroma_db"
TOP_K = 3

def main():
    # 1. Check if DB exists
    if not os.path.exists(PERSIST_DIR):
        print(f"Error: Database directory '{PERSIST_DIR}' not found. Run build_db.py first.")
        sys.exit(1)

    print("Loading ChromaDB Database...")
    client = PersistentClient(path=PERSIST_DIR)
    
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        print(f"Error loading database: {e}")
        sys.exit(1)

    print("\n" + "="*50)
    print("🤖 RAGCraft Engine (Terminal Mode)")
    print("="*50)
    print("Type 'exit' to quit.\n")

    # 2. Main Chat Loop
    while True:
        query = input("Ask me a question: ").strip()
        if query.lower() in ['exit', 'quit']:
            break
        if not query:
            continue

        try:
            # Step A: Embed the user's question
            query_embed_response = ollama.embed(model=EMBEDDING_MODEL, input=query)
            query_vector = query_embed_response['embeddings'][0]

            # Step B: Retrieve top chunks from ChromaDB
            results = collection.query(
                query_embeddings=[query_vector],
                n_results=TOP_K
            )
            
            documents = results['documents'][0]
            if not documents:
                print("I don't know based on the given context.\n")
                continue

            # Step C: Build the prompt with the strict context
            context_string = "\n\n".join(documents)
            system_prompt = (
                "You are a helpful assistant. Use ONLY the following pieces of context to answer the user's question. "
                "If the context does not contain the answer, reply exactly with: 'I don't know based on the given context.' "
                "Do NOT fabricate facts.\n\n"
                f"Context:\n{context_string}"
            )

            # Step D: Stream the response from the LLM
            print("\nThinking...")
            response = ollama.generate(
                model=LANGUAGE_MODEL,
                prompt=query,
                system=system_prompt,
                stream=True
            )
            
            print("Answer: ", end="", flush=True)
            for chunk in response:
                print(chunk['response'], end="", flush=True)
            print("\n" + "-"*50 + "\n")
            
        except Exception as e:
            print(f"\n[!] An error occurred: {e}\n")

if __name__ == "__main__":
    main()