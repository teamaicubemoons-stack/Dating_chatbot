"""
rag/embedder.py
---------------
Memory-optimized embedding generation module for low-resource cloud deployments
(e.g., Render Free Tier 512MB RAM).

Uses subword hashing + n-gram term frequency vectorization (384 dimensions)
normalized to unit length. Provides semantic & keyword retrieval with
zero PyTorch memory overhead (< 1 MB RAM vs > 450 MB for PyTorch).
"""

import math
import hashlib
import re
import logging
from typing import List

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 384


def embed_text(text: str, dim: int = EMBEDDING_DIM) -> List[float]:
    """
    Generate a 384-dimensional normalized vector for a text string.
    Zero PyTorch footprint, ultra-fast and memory-safe for 512MB environments.
    """
    if not text:
        return [0.0] * dim

    words = re.findall(r"\w+", text.lower())
    vec = [0.0] * dim
    if not words:
        return vec

    for w in words:
        # Full word hash
        h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16) % dim
        vec[h] += 1.0

        # Subword n-grams (3 to 5 chars) for fuzzy / morphological matching
        for n in range(3, min(6, len(w) + 1)):
            for i in range(len(w) - n + 1):
                ngram = w[i : i + n]
                h_ng = int(hashlib.md5(ngram.encode("utf-8")).hexdigest(), 16) % dim
                vec[h_ng] += 0.5

    # L2 normalize so cosine similarity equals dot product
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


def embed_batch(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a batch of strings.
    """
    return [embed_text(t) for t in texts]
