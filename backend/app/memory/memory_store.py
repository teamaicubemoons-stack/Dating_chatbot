"""
memory/memory_store.py
----------------------
Database persistence layer for conversation history, long-term memories,
rolling conversation summaries, and conversational state.

Responsibilities:
  - Save user and assistant messages to the SQLite Messages table.
  - Retrieve recent and full conversation history.
  - Manage UserMemory entities with strict user ownership and profile isolation.
  - Manage ConversationSummary and ConversationState.
  - Record audit logs for all memory lifecycle changes.
  - Provide GDPR/privacy export and deletion capabilities.
"""

import logging
import uuid
import json
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple

from sqlalchemy.orm import Session
from sqlalchemy import or_, and_

from app.models.db_models import (
    Message, User, Profile, UserMemory, ConversationSummary,
    ConversationState, MemoryAuditLog
)
from app.config.settings import settings

logger = logging.getLogger(__name__)


def ensure_user(db: Session, user_id: str) -> User:
    """Return User row for user_id, creating it if absent (idempotent)."""
    user = db.get(User, user_id)
    if user is None:
        user = User(id=user_id)
        db.add(user)
        db.commit()
        db.refresh(user)
        logger.debug("Created new user record: %s", user_id)
    return user


def ensure_profile(db: Session, profile_id: str, name: str = "", short_bio: str = "") -> Profile:
    """Return Profile row for profile_id, creating it if absent."""
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
    """Persist a single message ('user' or 'assistant') to the DB."""
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
    Returns messages in chronological order (oldest first).
    """
    limit = (n_turns or settings.max_recent_history_turns) * 2

    rows = (
        db.query(Message)
        .filter(Message.user_id == user_id, Message.profile_id == profile_id)
        .order_by(Message.created_at.desc(), Message.id.desc())
        .limit(limit)
        .all()
    )

    rows = list(reversed(rows))
    return [
        {"role": row.role, "content": row.content.strip()}
        for row in rows
        if row.content and row.content.strip()
    ]


def get_full_history(
    db: Session,
    user_id: str,
    profile_id: str,
) -> List[Message]:
    """Return all messages for a user+profile thread in chronological order."""
    return (
        db.query(Message)
        .filter(Message.user_id == user_id, Message.profile_id == profile_id)
        .order_by(Message.created_at.asc())
        .all()
    )


# ===========================================================================
# UserMemory Operations
# ===========================================================================

def save_user_memory(
    db: Session,
    user_id: str,
    memory_type: str,
    memory_text: str,
    profile_id: Optional[str] = None,
    normalized_key: Optional[str] = None,
    importance: float = 0.5,
    confidence: float = 1.0,
    source_type: str = "explicit",
    expires_at: Optional[datetime] = None,
    source_message_id: Optional[int] = None,
) -> UserMemory:
    """
    Persist or update a user memory.
    If a normalized_key exists, deduplicates or updates the existing active memory.
    """
    ensure_user(db, user_id)
    if profile_id:
        ensure_profile(db, profile_id)

    now = datetime.utcnow()
    existing: Optional[UserMemory] = None

    if normalized_key:
        query = db.query(UserMemory).filter(
            UserMemory.user_id == user_id,
            UserMemory.normalized_key == normalized_key,
            UserMemory.status == "active",
        )
        if memory_type == "relationship":
            query = query.filter(UserMemory.profile_id == profile_id)
        existing = query.first()

    if existing:
        # Update existing memory with confirmed information
        existing.memory_text = memory_text
        existing.importance = max(existing.importance, importance)
        existing.confidence = max(existing.confidence, confidence)
        existing.last_confirmed_at = now
        existing.updated_at = now
        existing.source_type = source_type
        if expires_at:
            existing.expires_at = expires_at
        db.commit()
        db.refresh(existing)
        _log_memory_audit(db, user_id, existing.id, "update", f"Updated key '{normalized_key}'")
        return existing

    # Create new memory
    mem = UserMemory(
        id=str(uuid.uuid4()),
        user_id=user_id,
        profile_id=profile_id,
        memory_type=memory_type,
        memory_text=memory_text,
        normalized_key=normalized_key,
        importance=importance,
        confidence=confidence,
        source_type=source_type,
        source_message_id=source_message_id,
        valid_from=now,
        expires_at=expires_at,
        last_confirmed_at=now,
        status="active",
        created_at=now,
        updated_at=now,
    )
    db.add(mem)
    db.commit()
    db.refresh(mem)
    _log_memory_audit(db, user_id, mem.id, "create", f"Created memory of type '{memory_type}'")
    return mem


def get_active_user_memories(
    db: Session,
    user_id: str,
    profile_id: Optional[str] = None,
    memory_types: Optional[List[str]] = None,
) -> List[UserMemory]:
    """
    Retrieve all active, non-expired memories accessible to the current turn.
    Rules:
      - Memory MUST belong to user_id (strict user isolation).
      - Relationship memories are STRICTLY filtered to profile_id.
      - General facts and preferences are shared across personas.
    """
    now = datetime.utcnow()

    # Scope condition: (global memories where profile_id is None) OR (memories specifically for this profile)
    scope_cond = or_(
        UserMemory.profile_id.is_(None),
        UserMemory.profile_id == profile_id,
    ) if profile_id else UserMemory.profile_id.is_(None)

    query = (
        db.query(UserMemory)
        .filter(
            UserMemory.user_id == user_id,
            UserMemory.status == "active",
            scope_cond,
            or_(UserMemory.expires_at.is_(None), UserMemory.expires_at > now),
        )
    )

    if memory_types:
        query = query.filter(UserMemory.memory_type.in_(memory_types))

    return query.order_by(UserMemory.importance.desc(), UserMemory.created_at.desc()).all()


def update_user_memory(
    db: Session,
    memory_id: str,
    user_id: str,
    memory_text: Optional[str] = None,
    importance: Optional[float] = None,
    status: Optional[str] = None,
) -> Optional[UserMemory]:
    """Update memory fields with verified user ownership."""
    mem = db.query(UserMemory).filter(
        UserMemory.id == memory_id,
        UserMemory.user_id == user_id,
    ).first()

    if not mem:
        return None

    if memory_text is not None:
        mem.memory_text = memory_text
    if importance is not None:
        mem.importance = importance
    if status is not None:
        mem.status = status

    mem.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(mem)
    _log_memory_audit(db, user_id, memory_id, "patch", "User updated memory")
    return mem


def delete_user_memory(db: Session, memory_id: str, user_id: str) -> bool:
    """Soft-delete a user memory (mark as status='deleted') with ownership validation."""
    mem = db.query(UserMemory).filter(
        UserMemory.id == memory_id,
        UserMemory.user_id == user_id,
    ).first()

    if not mem:
        return False

    mem.status = "deleted"
    mem.updated_at = datetime.utcnow()
    db.commit()
    _log_memory_audit(db, user_id, memory_id, "delete", "User deleted memory")
    return True


def forget_memories(
    db: Session,
    user_id: str,
    scope: str = "all",
    profile_id: Optional[str] = None,
) -> int:
    """
    Bulk delete memories according to scope:
      - 'all'             : all memories for this user
      - 'relationship'    : relationship memories for this user and specified profile_id
      - 'profile_facts'   : profile_fact and preference memories
    """
    query = db.query(UserMemory).filter(
        UserMemory.user_id == user_id,
        UserMemory.status == "active",
    )

    if scope == "relationship":
        if profile_id:
            query = query.filter(UserMemory.profile_id == profile_id)
        else:
            query = query.filter(UserMemory.memory_type == "relationship")
    elif scope == "profile_facts":
        query = query.filter(UserMemory.memory_type.in_(["profile_fact", "preference"]))

    memories = query.all()
    count = len(memories)
    for mem in memories:
        mem.status = "deleted"
        mem.updated_at = datetime.utcnow()

    db.commit()
    _log_memory_audit(db, user_id, None, "forget", f"Bulk forgot scope '{scope}' ({count} records)")
    return count


def _log_memory_audit(db: Session, user_id: str, memory_id: Optional[str], action: str, reason: str = ""):
    """Append entry to memory audit log."""
    try:
        audit = MemoryAuditLog(
            user_id=user_id,
            memory_id=memory_id,
            action=action,
            reason=reason,
            created_at=datetime.utcnow(),
        )
        db.add(audit)
        db.commit()
    except Exception as exc:
        logger.warning("Failed to record memory audit log: %s", exc)


# ===========================================================================
# ConversationSummary Operations
# ===========================================================================

def get_latest_summary(db: Session, user_id: str, profile_id: str) -> Optional[ConversationSummary]:
    """Retrieve the latest rolling conversation summary for this user and persona."""
    return (
        db.query(ConversationSummary)
        .filter(
            ConversationSummary.user_id == user_id,
            ConversationSummary.profile_id == profile_id,
        )
        .order_by(ConversationSummary.version.desc())
        .first()
    )


def save_summary(
    db: Session,
    user_id: str,
    profile_id: str,
    summary_text: str,
    start_id: Optional[int] = None,
    end_id: Optional[int] = None,
) -> ConversationSummary:
    """Store or update the rolling summary for a user+profile pair."""
    latest = get_latest_summary(db, user_id, profile_id)
    version = (latest.version + 1) if latest else 1

    summary = ConversationSummary(
        user_id=user_id,
        profile_id=profile_id,
        summary_text=summary_text,
        source_message_start_id=start_id,
        source_message_end_id=end_id,
        version=version,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    db.add(summary)
    db.commit()
    db.refresh(summary)
    return summary


# ===========================================================================
# ConversationState Operations
# ===========================================================================

def get_or_create_conversation_state(db: Session, user_id: str, profile_id: str) -> ConversationState:
    """Fetch or initialize the current conversation state for a user+profile pair."""
    state = (
        db.query(ConversationState)
        .filter(
            ConversationState.user_id == user_id,
            ConversationState.profile_id == profile_id,
        )
        .first()
    )
    if state is None:
        state = ConversationState(
            user_id=user_id,
            profile_id=profile_id,
            current_topic="",
            conversation_tone="casual",
            user_stated_emotion="",
            inferred_emotional_cue="",
            open_threads_json="[]",
            updated_at=datetime.utcnow(),
        )
        db.add(state)
        db.commit()
        db.refresh(state)
    return state


def update_conversation_state(
    db: Session,
    user_id: str,
    profile_id: str,
    current_topic: Optional[str] = None,
    conversation_tone: Optional[str] = None,
    user_stated_emotion: Optional[str] = None,
    inferred_emotional_cue: Optional[str] = None,
    open_threads: Optional[List[str]] = None,
    last_follow_up_at: Optional[datetime] = None,
    recent_context: Optional[Dict[str, Any]] = None,
) -> ConversationState:
    """Update active conversational cues, open topics, and context in the session state."""
    state = get_or_create_conversation_state(db, user_id, profile_id)

    if current_topic is not None:
        state.current_topic = current_topic
    if conversation_tone is not None:
        state.conversation_tone = conversation_tone
    if user_stated_emotion is not None:
        state.user_stated_emotion = user_stated_emotion
    if inferred_emotional_cue is not None:
        state.inferred_emotional_cue = inferred_emotional_cue
    if open_threads is not None:
        state.open_threads_json = json.dumps(open_threads)
    if last_follow_up_at is not None:
        state.last_follow_up_at = last_follow_up_at
    if recent_context is not None:
        existing_ctx = {}
        try:
            existing_ctx = json.loads(state.recent_context_json or "{}")
        except Exception:
            existing_ctx = {}
        existing_ctx.update(recent_context)
        state.recent_context_json = json.dumps(existing_ctx)

    state.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(state)
    return state


# ===========================================================================
# Privacy & GDPR Operations
# ===========================================================================

def export_user_data(db: Session, user_id: str) -> Dict[str, Any]:
    """Export all messages, memories, and state for the requested user."""
    memories = db.query(UserMemory).filter(UserMemory.user_id == user_id).all()
    summaries = db.query(ConversationSummary).filter(ConversationSummary.user_id == user_id).all()
    messages = db.query(Message).filter(Message.user_id == user_id).all()
    states = db.query(ConversationState).filter(ConversationState.user_id == user_id).all()

    return {
        "user_id": user_id,
        "exported_at": datetime.utcnow(),
        "memories": memories,
        "summaries": summaries,
        "messages": messages,
        "states": [
            {
                "profile_id": s.profile_id,
                "current_topic": s.current_topic,
                "conversation_tone": s.conversation_tone,
                "user_stated_emotion": s.user_stated_emotion,
                "open_threads": json.loads(s.open_threads_json or "[]"),
                "updated_at": s.updated_at,
            }
            for s in states
        ],
    }


def delete_user_account_data(db: Session, user_id: str) -> Dict[str, int]:
    """Completely erase all messages, memories, state, and user records for user_id."""
    num_messages = db.query(Message).filter(Message.user_id == user_id).delete()
    num_memories = db.query(UserMemory).filter(UserMemory.user_id == user_id).delete()
    num_summaries = db.query(ConversationSummary).filter(ConversationSummary.user_id == user_id).delete()
    num_states = db.query(ConversationState).filter(ConversationState.user_id == user_id).delete()
    num_users = db.query(User).filter(User.id == user_id).delete()

    db.commit()
    _log_memory_audit(db, user_id, None, "account_delete", "User permanently erased account data")

    return {
        "messages": num_messages,
        "memories": num_memories,
        "summaries": num_summaries,
        "states": num_states,
        "users": num_users,
    }
