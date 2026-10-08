"""
memory/conversation_memory.py
------------------------------
High-level memory management for per-user, per-profile conversations.

Responsibilities:
  1. Orchestrate fetching recent conversation history from memory_store.
  2. Decide when a conversation fact is "significant" enough to push to
     the vector store for long-term RAG memory (simple heuristic approach).
  3. Extract and store user-stated facts (name, preferences, key life details)
     into ChromaDB via retriever.store_memory_fact() for future recall.

This module sits between chat_engine.py and memory_store.py/retriever.py,
keeping the logic for "what to remember and how" in one place.
"""

import logging
import re
import uuid
from typing import List, Dict

from sqlalchemy.orm import Session

from app.memory.memory_store import get_recent_history, save_message, ensure_user, ensure_profile
from app.rag.retriever import store_memory_fact
from app.config.settings import settings

logger = logging.getLogger(__name__)

# Simple patterns that indicate the user shared a memorable personal fact
_MEMORY_TRIGGERS = [
    r"\bmy name is\b",
    r"\bi('m| am)\b.{0,30}\b(years old|yr)\b",
    r"\bi (love|hate|like|enjoy|prefer|work|live|study)\b",
    r"\bmy (job|work|career|major|hobby|pet|partner|wife|husband|boyfriend|girlfriend)\b",
    r"\bi('m| am) from\b",
    r"\bmy favorite\b",
]
_MEMORY_PATTERN = re.compile("|".join(_MEMORY_TRIGGERS), re.IGNORECASE)


def get_conversation_history(
    db: Session,
    user_id: str,
    profile_id: str,
) -> List[Dict[str, str]]:
    """
    Retrieve recent conversation history for injection into the LLM prompt.

    Returns the last N turns as [{"role": ..., "content": ...}] list.
    N is controlled by settings.max_recent_history_turns.
    """
    return get_recent_history(db, user_id, profile_id)


def persist_exchange(
    db: Session,
    user_id: str,
    profile_id: str,
    user_message: str,
    ai_reply: str,
) -> None:
    """
    Save both sides of a conversation exchange to the DB,
    and trigger long-term memory storage for significant user facts.

    Args:
        db:           Active SQLAlchemy session.
        user_id:      The user's ID.
        profile_id:   The active persona's ID.
        user_message: What the user said.
        ai_reply:     What the AI responded.
    """
    # Ensure user and profile rows exist
    ensure_user(db, user_id)

    # Persist both messages
    save_message(db, user_id, profile_id, "user", user_message)
    save_message(db, user_id, profile_id, "assistant", ai_reply)

    # Check if the user shared a memorable fact worth indexing for RAG
    _maybe_store_memory_fact(profile_id, user_id, user_message)


def _maybe_store_memory_fact(profile_id: str, user_id: str, user_message: str) -> None:
    """
    Heuristically detect if the user shared a personal fact and store it
    in the vector store for long-term retrieval.

    Conservative approach: only store if message matches known patterns.
    In production, this could be replaced with an LLM-based extractor.
    """
    if len(user_message) < 10:
        return  # Too short to contain a meaningful fact

    if _MEMORY_PATTERN.search(user_message):
        fact_id = f"{profile_id}_{user_id}_{uuid.uuid4().hex[:8]}"
        logger.debug("Storing memory fact: '%s...'", user_message[:50])
        store_memory_fact(
            profile_id=profile_id,
            user_id=user_id,
            fact=user_message,
            fact_id=fact_id,
        )
