"""
models/db_models.py
-------------------
SQLAlchemy ORM models. Defines the full database schema:
  - User                 : registered or anonymous users
  - Profile              : AI persona identifiers (synced from JSON configs)
  - Message              : individual chat messages per user+profile thread
  - UserMemory           : long-term user preferences, profile facts, episodic events, and relationship memories
  - ConversationSummary  : rolling conversation-level factual summaries
  - ConversationState    : immediate conversation state, topic tracking, open threads, and emotional cues
  - MemoryAuditLog       : audit record for memory creation, modification, deletion, and privacy actions
  - LLMStatus            : single-row table tracking active LLM provider + last health check

All tables are created automatically at startup via `Base.metadata.create_all`.
"""

import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Float, Text, DateTime, Boolean, ForeignKey, Index
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True)           # UUID or anonymous session id
    created_at = Column(DateTime, default=datetime.utcnow)

    messages = relationship("Message", back_populates="user", cascade="all, delete-orphan")
    memories = relationship("UserMemory", back_populates="user", cascade="all, delete-orphan")
    summaries = relationship("ConversationSummary", back_populates="user", cascade="all, delete-orphan")
    states = relationship("ConversationState", back_populates="user", cascade="all, delete-orphan")


class Profile(Base):
    """
    Represents a loaded persona. Kept in sync with persona JSON configs at startup.
    Storing it in DB lets us join against messages and enables future admin features.
    """
    __tablename__ = "profiles"

    id = Column(String, primary_key=True)           # matches persona JSON "id" field
    name = Column(String, nullable=False)
    short_bio = Column(Text, default="")
    avatar_url = Column(String, default="")
    created_at = Column(DateTime, default=datetime.utcnow)

    messages = relationship("Message", back_populates="profile", cascade="all, delete-orphan")
    memories = relationship("UserMemory", back_populates="profile", cascade="all, delete-orphan")
    summaries = relationship("ConversationSummary", back_populates="profile", cascade="all, delete-orphan")
    states = relationship("ConversationState", back_populates="profile", cascade="all, delete-orphan")


class Message(Base):
    """Stores every chat turn — both user messages and AI replies."""
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    profile_id = Column(String, ForeignKey("profiles.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String, nullable=False)           # "user" | "assistant"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    user = relationship("User", back_populates="messages")
    profile = relationship("Profile", back_populates="messages")


class UserMemory(Base):
    """
    Persistent long-term memory entity per user and optional persona scope.
    Memory types:
      - 'profile_fact'          : stable user facts (name, job, hometown, education)
      - 'preference'            : user likes/dislikes (favorite food, music, texting style)
      - 'episodic'              : discrete life events (interviews, trips, exams, milestones)
      - 'relationship'          : persona-specific shared jokes, nicknames, threads (strictly isolated per profile_id)
      - 'ongoing_topic'         : current active discussions needing follow-up
      - 'conversation_pattern'  : learned high-level conversational style preferences
    """
    __tablename__ = "user_memories"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    profile_id = Column(String, ForeignKey("profiles.id", ondelete="SET NULL"), nullable=True, index=True)
    memory_type = Column(String(50), nullable=False, index=True)
    memory_text = Column(Text, nullable=False)
    normalized_key = Column(String(150), nullable=True, index=True)
    source_message_id = Column(Integer, ForeignKey("messages.id", ondelete="SET NULL"), nullable=True)
    confidence = Column(Float, default=1.0)
    importance = Column(Float, default=0.5)
    source_type = Column(String(20), default="explicit")  # "explicit" | "inferred"
    valid_from = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True, index=True)
    last_confirmed_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String(20), default="active", index=True)  # "active" | "deleted" | "archived"
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="memories")
    profile = relationship("Profile", back_populates="memories")

    __table_args__ = (
        Index("ix_user_memories_lookup", "user_id", "status", "memory_type"),
        Index("ix_user_profile_memories", "user_id", "profile_id", "status"),
    )


class ConversationSummary(Base):
    """
    Rolling summary of extended conversations per user+profile thread.
    Preserves historical context beyond the recent 10-turn window.
    """
    __tablename__ = "conversation_summaries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    profile_id = Column(String, ForeignKey("profiles.id", ondelete="CASCADE"), nullable=False, index=True)
    conversation_id = Column(String(100), nullable=True, index=True)
    summary_text = Column(Text, nullable=False)
    source_message_start_id = Column(Integer, nullable=True)
    source_message_end_id = Column(Integer, nullable=True)
    version = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="summaries")
    profile = relationship("Profile", back_populates="summaries")

    __table_args__ = (
        Index("ix_conv_summary_user_profile", "user_id", "profile_id"),
    )


class ConversationState(Base):
    """
    Ephemeral conversational state maintaining immediate topic, emotional tone,
    and open follow-up threads per user+profile session.
    """
    __tablename__ = "conversation_state"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    profile_id = Column(String, ForeignKey("profiles.id", ondelete="CASCADE"), nullable=False, index=True)
    current_topic = Column(String(200), default="")
    conversation_tone = Column(String(100), default="casual")
    user_stated_emotion = Column(String(100), default="")
    inferred_emotional_cue = Column(String(100), default="")
    open_threads_json = Column(Text, default="[]")
    recent_context_json = Column(Text, default="{}")
    last_follow_up_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="states")
    profile = relationship("Profile", back_populates="states")

    __table_args__ = (
        Index("ix_conv_state_user_profile", "user_id", "profile_id", unique=True),
    )


class MemoryAuditLog(Base):
    """
    Auditing table tracking all memory life-cycle events (create, patch, delete, forget)
    ensuring privacy compliance without logging raw sensitive message content.
    """
    __tablename__ = "memory_audit_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, nullable=False, index=True)
    memory_id = Column(String, nullable=True, index=True)
    action = Column(String(50), nullable=False)  # "create" | "update" | "delete" | "forget_all"
    reason = Column(String(200), default="")
    created_at = Column(DateTime, default=datetime.utcnow)


class LLMStatus(Base):
    """
    Single-row table that tracks which LLM provider is currently active.
    Row with id=1 is the canonical status record — created at startup if absent.
    """
    __tablename__ = "llm_status"

    id = Column(Integer, primary_key=True, default=1)
    current_provider = Column(String, default="groq")   # "groq" | "claude"
    last_checked_at = Column(DateTime, default=datetime.utcnow)
    groq_available = Column(Boolean, default=True)
