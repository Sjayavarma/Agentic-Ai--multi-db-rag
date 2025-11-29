# config.py

# Embedding model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Ollama model
GEMMA_MODEL_NAME = "gemma:2b"

# FAISS configuration
FAISS_INDEX_PATH = "data/faiss.index"   # <-- REQUIRED for dbs.py import
TOP_K = 3
