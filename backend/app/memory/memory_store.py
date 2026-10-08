"""
memory/memory_store.py
----------------------
Database persistence layer for conversation history.

Responsibilities:
  - Save user and assistant messages to the SQLite Messages table.
  - Retrieve recent conversation history for a user+profile thread.
  - Ensure/create user and profile rows as needed (auto-registration).
  - Optionally push important conversation facts to the vector store for RAG.

This module is the only file that performs direct SQLAlchemy ORM operations
on Message, User, and Profile tables. Keep all DB logic contained here.
"""

import logging
import uuid
from datetime import datetime
from typing import List, Dict, Optional

from sqlalchemy.orm import Session

from app.models.db_models import Message, User, Profile
from app.config.settings import settings

logger = logging.getLogger(__name__)


def ensure_user(db: Session, user_id: str) -> User:
    """
    Return the User row for user_id, creating it if it doesn't exist.
    Safe to call on every request — idempotent.
    """
    user = db.get(User, user_id)
    if user is None:
        user = User(id=user_id)
        db.add(user)
        db.commit()
        db.refresh(user)
        logger.debug("Created new user record: %s", user_id)
    return user


def ensure_profile(db: Session, profile_id: str, name: str = "", short_bio: str = "") -> Profile:
    """
    Return the Profile row for profile_id, creating it if it doesn't exist.
    Called during startup persona sync and on first chat with a profile.
    """
    profile = db.get(Profile, profile_id)
    if profile is None:
        profile = Profile(id=profile_id, name=name, short_bio=short_bio)
        db.add(profile)
        db.commit()
        db.refresh(profile)
        logger.debug("Created profile record: %s", profile_id)
    return profile


def save_message(
    db: Session,
    user_id: str,
    profile_id: str,
    role: str,
    content: str,
) -> Message:
    """
    Persist a single message (either 'user' or 'assistant') to the DB.

    Args:
        db:         Active SQLAlchemy session.
        user_id:    The user's ID string.
        profile_id: The active persona's ID.
        role:       "user" or "assistant".
        content:    The message text.

    Returns:
        The newly created Message ORM object.
    """
    msg = Message(
        user_id=user_id,
        profile_id=profile_id,
        role=role,
        content=content,
        created_at=datetime.utcnow(),
    )
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return msg


def get_recent_history(
    db: Session,
    user_id: str,
    profile_id: str,
    n_turns: Optional[int] = None,
) -> List[Dict[str, str]]:
    """
    Retrieve the N most recent conversation turns for a user+profile pair.

    Returns messages in chronological order (oldest first), formatted as
    the standard [{"role": ..., "content": ...}] list for LLM consumption.

    Args:
        db:         Active SQLAlchemy session.
        user_id:    The user's ID.
        profile_id: The persona's ID.
        n_turns:    Number of recent turns to fetch. Uses settings default if None.

    Returns:
        List of message dicts, oldest-first.
    """
    limit = (n_turns or settings.max_recent_history_turns) * 2  # *2: both sides of each turn

    rows = (
        db.query(Message)
        .filter(Message.user_id == user_id, Message.profile_id == profile_id)
        .order_by(Message.created_at.desc())
        .limit(limit)
        .all()
    )

    # Reverse to get chronological order (oldest first for LLM context)
    rows = list(reversed(rows))
    return [{"role": row.role, "content": row.content} for row in rows]


def get_full_history(
    db: Session,
    user_id: str,
    profile_id: str,
) -> List[Message]:
    """Return all messages for a user+profile thread, chronological order."""
    return (
        db.query(Message)
        .filter(Message.user_id == user_id, Message.profile_id == profile_id)
        .order_by(Message.created_at.asc())
        .all()
    )


def generate_user_id() -> str:
    """Generate a new anonymous user session ID."""
    return str(uuid.uuid4())
