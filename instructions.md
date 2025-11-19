# RAG Pipeline Improvement & Troubleshooting Advisor

This document outlines the improvements and best practices implemented in the RAG pipeline, based on the provided guidelines.

## 1. Embedding Response Handling

**Rule:** Always verify the shape of the response from `ollama.embed`. If errors occur, print or log the entire raw response object. Provide a unified parsing method.

**Implementation:**
- The `embed_batch` function in `build_db.py` and `embed_query` function in `rag_app.py` now include a unified parsing logic to handle various response structures from `ollama.embed` (e.g., `resp["embeddings"]`, `resp.embedding`, `resp.data[i].embedding`, `list-of-dicts`).
- Robust `try-except` blocks are used to catch embedding errors, printing the raw error message and the shape of the data being processed for debugging.
- Unexpected response structures are logged with the raw response object.

## 2. ID Generation for Documents

**Rule:** Avoid ID collisions between batches. Use deterministic IDs (e.g., `doc_0`, `doc_1`) or UUID-based IDs for total uniqueness. Ensure IDs remain stable across rebuilds unless intentionally reset.

**Implementation:**
- In `build_db.py`, document IDs are now generated using `uuid.uuid4()`. This ensures total uniqueness across batches and rebuilds, preventing potential collisions.

## 3. Rebuild Logic

**Rule:** If a vector database already exists: warn the user, provide options (Skip, Rebuild, Rebuild only new files), or accept a `--force` parameter.

**Implementation:**
- The `main` function in `build_db.py` now checks if the `PERSIST_DIR` exists and is non-empty.
- It uses `argparse` to accept a `--force` command-line argument. If `--force` is present, the existing database is automatically deleted and rebuilt.
- If `--force` is not used, the user is prompted with options to `(S)kip` or `(R)ebuild`. The "New files only" option is acknowledged but noted as not implemented for this iteration.
- `shutil.rmtree(PERSIST_DIR)` is used for safe deletion of the existing database directory during a rebuild.

## 4. Chunking Strategy

**Rule:** Avoid single-line chunks unless the document is tiny. Provide improved chunking options (sentence-based, paragraph-based, sliding window with overlap) and explain pros/cons.

**Implementation:**
- In `build_db.py`, the `read_dataset` function now uses `nltk.sent_tokenize` for sentence-based chunking. This provides better contextual chunks than single-line chunks.
- NLTK's `punkt` tokenizer data is automatically downloaded if not already present.
- Comments in the code explain that for more advanced scenarios, sliding window chunking with overlap could be considered.

## 5. Similarity Scoring

**Rule:** Convert distances to a similarity score only when helpful. Also return raw distances for debugging. Explain how different distance metrics affect retrieval.

**Implementation:**
- In `rag_app.py`, the `retrieve` function now returns a dictionary for each retrieved item, including `chunk`, `distance`, `similarity`, and `metadata`.
- A heuristic `1.0 / (1.0 + distance)` is used to convert raw distance (assuming L2/Euclidean) into a similarity score, where a higher score indicates greater similarity.
- The `main` function prints both the similarity and raw distance for each retrieved chunk.
- An "Explanation of Similarity Scoring" section is added to the output, detailing Cosine, Euclidean (L2), and L2 Normalized Euclidean distances and how they relate to similarity.

## 6. Hallucination Prevention

**Rule:** When generating answers: use ONLY the provided retrieved context. If the context does not contain the answer, reply with: "I don’t know based on the given context." Never fabricate facts.

**Implementation:**
- The `build_instruction_prompt` function in `rag_app.py` has been updated with a strict system prompt:
    - "Use ONLY the following pieces of context to answer the question."
    - "If the context does not contain the answer, reply with: 'I don't know based on the given context.'"
    - "Do NOT fabricate facts. You may summarize, combine pieces, or explain—but not invent."
- If `retrieve` returns no relevant documents, the `main` function in `rag_app.py` now explicitly prints "I don't know based on the given context."

## 7. Batch Size & Rate Control

**Rule:** If embedding large documents: use batch sizes of 32–128 depending on memory. If using a remote Ollama server, throttle requests. Provide dynamic batching suggestions.

**Implementation:**
- In `build_db.py`, `BATCH_SIZE` is set to `64` with a comment indicating recommended ranges (32-128) based on memory and model size.
- A commented-out `time.sleep(0.1)` is included in `build_db.py` as a suggestion for rate control when interacting with remote Ollama servers.

## 8. Debugging Help

**Rule:** If anything fails, ALWAYS: print the shape of the data being processed, print the raw error message, provide steps to fix it, suggest alternative code snippets.

**Implementation:**
- Extensive `DEBUG` print statements have been added throughout both `build_db.py` and `rag_app.py` within `try-except` blocks.
- These debug messages include:
    - The raw error message (`e`).
    - The shape of the data being processed (e.g., `len(texts)`).
    - Actionable suggestions for fixing the issue (e.g., "Check if Ollama server is running and EMBEDDING_MODEL is correct").
    - Alternative approaches (e.g., "Try a smaller batch size or a different embedding model").
