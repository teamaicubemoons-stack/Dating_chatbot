"""
memory/conversation_memory.py
------------------------------
High-level memory orchestration for per-user, per-profile conversations.

Responsibilities:
  1. Fetch recent conversation turns from memory_store.
  2. Persist user and assistant messages to database.
"""

import logging
from typing import List, Dict, Tuple
from sqlalchemy.orm import Session

from app.memory.memory_store import get_recent_history, save_message, ensure_user, ensure_profile
from app.models.db_models import Message
from app.config.settings import settings

logger = logging.getLogger(__name__)


def get_conversation_history(
    db: Session,
    user_id: str,
    profile_id: str,
) -> List[Dict[str, str]]:
    """
    Retrieve recent conversation history for injection into the LLM prompt.
    Returns the last N turns as [{"role": ..., "content": ...}] list.
    """
    return get_recent_history(db, user_id, profile_id)


def persist_exchange(
    db: Session,
    user_id: str,
    profile_id: str,
    user_message: str,
    ai_reply: str,
) -> Tuple[Message, Message]:
    """
    Save both sides of a conversation exchange to the DB.
    Returns: (user_message_record, assistant_message_record)
    """
    ensure_user(db, user_id)

    # Persist both messages
    user_msg = save_message(db, user_id, profile_id, "user", user_message)
    asst_msg = save_message(db, user_id, profile_id, "assistant", ai_reply)

    return user_msg, asst_msg
