"""
rag/embedder.py
---------------
Embedding generation module. Converts text → dense vector representations
for storage and similarity search in ChromaDB.

Uses sentence-transformers locally (no extra API call, no cost, fast on CPU).
The model is loaded once at module import via a singleton pattern to avoid
re-loading on every request.

To swap the embedding model: change EMBEDDING_MODEL below — nothing else changes.
"""

import logging
from typing import List, Union

logger = logging.getLogger(__name__)

# The embedding model used for all vector operations.
# 'all-MiniLM-L6-v2' is fast, lightweight, and high quality for semantic similarity.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_model = None  # Lazy-loaded singleton


def _get_model():
    """Lazily load the sentence-transformer model (only on first call)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        _model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model loaded successfully")
    return _model


def embed_text(text: str) -> List[float]:
    """
    Generate a single embedding vector for a text string.

    Args:
        text: The input string to embed.

    Returns:
        A list of floats representing the embedding vector.
    """
    model = _get_model()
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()


def embed_batch(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a batch of strings (more efficient than one-by-one).

    Args:
        texts: List of input strings.

    Returns:
        List of embedding vectors (same order as input).
    """
    if not texts:
        return []
    model = _get_model()
    embeddings = model.encode(texts, convert_to_numpy=True, batch_size=32)
    return [e.tolist() for e in embeddings]
