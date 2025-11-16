# RAGCraft

A minimal Retrieval-Augmented Generation (RAG) engine using **Ollama** for embeddings & LLM inference and **ChromaDB** for vector storage.  
Designed to be lightweight, persistent (document embeddings saved on disk), and easy to run locally.

---

## Features

- Build and persist a Chroma vector DB from plain text (`build_db.py`)
- Query the persisted DB and answer with an LLM using retrieval context (`rag_app.py`)
- Safe, robust handling of Ollama & Chroma response shapes
- Small demo dataset: `docs/cat-facts.txt`
- Scripts to help initialize environment

---

## Quick start

> Requirements: Python 3.10+, Ollama daemon (if using ollama), local git, and pip.

1. Clone repo
```bash
git clone https://github.com/AbhijeetAndhalkar/RAGCraft.git
cd RAGCraft
