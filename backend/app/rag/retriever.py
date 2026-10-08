"""
rag/retriever.py
----------------
High-level RAG retrieval interface used by chat_engine.py.

On each incoming message, retriever.py:
  1. Queries the persona's ChromaDB collection for relevant backstory/trait chunks.
  2. Queries for relevant long-term memory facts from past conversations
     (user preferences, topics mentioned, emotional moments).
  3. Deduplicates and formats results into a clean context string for prompt_builder.

The retriever is the ONLY module that calls vector_store.query() — keeping
RAG-specific logic isolated here, away from the chat engine.
"""

import logging
from typing import List

from app.rag.vector_store import query, add_texts
from app.config.settings import settings

logger = logging.getLogger(__name__)

# Weight: number of persona knowledge chunks vs. memory chunks to retrieve
_PERSONA_CHUNKS = 2
_MEMORY_CHUNKS = 3


def retrieve_context(
    profile_id: str,
    user_message: str,
    user_id: str,
) -> str:
    """
    Retrieve and format the most relevant RAG context for a chat turn.

    Combines:
      - Persona knowledge (backstory/traits relevant to the current topic)
      - Long-term conversation memory (user facts learned in past sessions)

    Args:
        profile_id:   Which persona's collection to search.
        user_message: Current user message (used as the search query).
        user_id:      User ID for filtering user-specific memory chunks.

    Returns:
        A formatted string of relevant context, or empty string if nothing found.
    """
    context_chunks: List[str] = []

    # --- 1. Persona knowledge retrieval ---
    try:
        persona_chunks = query(
            profile_id=profile_id,
            query_text=user_message,
            top_k=_PERSONA_CHUNKS,
            where={"type": {"$in": ["backstory", "trait"]}},
        )
        context_chunks.extend(persona_chunks)
        logger.debug("Retrieved %d persona knowledge chunks", len(persona_chunks))
    except Exception as exc:
        logger.warning("Persona RAG retrieval failed: %s", exc)

    # --- 2. Long-term memory retrieval (user-specific facts) ---
    try:
        memory_chunks = query(
            profile_id=profile_id,
            query_text=user_message,
            top_k=_MEMORY_CHUNKS,
            # ChromaDB 0.5+ requires $and for multiple filter conditions
            where={"$and": [{"type": {"$eq": "memory"}}, {"user_id": {"$eq": user_id}}]},
        )
        context_chunks.extend(memory_chunks)
        logger.debug("Retrieved %d memory chunks for user %s", len(memory_chunks), user_id)
    except Exception as exc:
        # Memory retrieval failures should NOT block the chat response
        logger.warning("Memory RAG retrieval failed: %s", exc)

    if not context_chunks:
        return ""

    # Deduplicate while preserving order
    seen = set()
    unique_chunks = []
    for chunk in context_chunks:
        if chunk not in seen:
            seen.add(chunk)
            unique_chunks.append(chunk)

    # Format as a clean, readable context block
    formatted = "\n".join(f"• {chunk}" for chunk in unique_chunks)
    return formatted


def store_memory_fact(
    profile_id: str,
    user_id: str,
    fact: str,
    fact_id: str,
) -> None:
    """
    Persist a distilled memory fact about the user into the vector store
    for future retrieval. Called by memory_store.py for significant conversation facts.

    Args:
        profile_id: The active persona's ID.
        user_id:    The user's ID (stored in metadata for filtering).
        fact:       The distilled fact text to store (e.g. "User's name is Alex").
        fact_id:    Unique ID for this fact (to prevent duplicates on re-index).
    """
    try:
        add_texts(
            profile_id=profile_id,
            texts=[fact],
            metadatas=[{"type": "memory", "user_id": user_id}],
            ids=[fact_id],
        )
        logger.debug("Stored memory fact for user %s in profile %s", user_id, profile_id)
    except Exception as exc:
        logger.warning("Failed to store memory fact: %s", exc)
