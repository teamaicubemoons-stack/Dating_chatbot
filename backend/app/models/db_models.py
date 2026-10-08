"""
models/db_models.py
-------------------
SQLAlchemy ORM models. Defines the full database schema:
  - User         : registered or anonymous users
  - Profile      : AI persona identifiers (synced from JSON configs)
  - Message      : individual chat messages per user+profile thread
  - LLMStatus    : single-row table tracking active LLM provider + last health check

All tables are created automatically at startup via `Base.metadata.create_all`.
"""

from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Text, DateTime, Boolean, ForeignKey
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True)           # UUID or anonymous session id
    created_at = Column(DateTime, default=datetime.utcnow)

    messages = relationship("Message", back_populates="user", cascade="all, delete-orphan")


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


class Message(Base):
    """Stores every chat turn — both user messages and AI replies."""
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    profile_id = Column(String, ForeignKey("profiles.id"), nullable=False)
    role = Column(String, nullable=False)           # "user" | "assistant"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="messages")
    profile = relationship("Profile", back_populates="messages")


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
