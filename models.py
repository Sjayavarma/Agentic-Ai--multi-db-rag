# models.py
"""
Holds model loaders:
- MiniLM embeddings
- (Optional) TinyLlama as agent LLM
- (Optional) Gemma-2B is loaded in gemma_generator.py
"""

from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_NAME


# Global embedding model (MiniLM)
_embedding_model = None


def get_embedding_model():
    """Load and cache MiniLM embedding model."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def embed_texts(texts):
    """
    Embed a list of strings using MiniLM.
    Returns a numpy array of shape (n, dim).
    """
    model = get_embedding_model()
    return model.encode(texts, convert_to_numpy=True)
