"""
tests/test_models_and_memory.py
--------------------------------
Unit tests for database models, memory persistence, user isolation,
and profile scoping.
"""

import pytest
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models.db_models import (
    Base, User, Profile, Message, UserMemory, ConversationSummary,
    ConversationState
)
from app.memory.memory_store import (
    save_user_memory, get_active_user_memories, update_user_memory,
    delete_user_memory, forget_memories, get_or_create_conversation_state,
    update_conversation_state, save_summary, get_latest_summary,
    export_user_data, delete_user_account_data
)


@pytest.fixture
def db():
    """In-memory SQLite database session for fast, isolated tests."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_user_memory_crud_and_deduplication(db):
    """Test creating, deduplicating, and updating a user memory."""
    # 1. Create initial memory
    mem1 = save_user_memory(
        db=db,
        user_id="user_123",
        memory_type="profile_fact",
        memory_text="User works as a Software Engineer",
        normalized_key="fact:occupation",
        importance=0.8,
    )
    assert mem1.id is not None
    assert mem1.status == "active"
    assert mem1.normalized_key == "fact:occupation"

    # 2. Add with same normalized_key should update rather than duplicate
    mem2 = save_user_memory(
        db=db,
        user_id="user_123",
        memory_type="profile_fact",
        memory_text="User works as a Lead AI Engineer",
        normalized_key="fact:occupation",
        importance=0.9,
    )
    assert mem1.id == mem2.id
    assert mem2.memory_text == "User works as a Lead AI Engineer"
    assert mem2.importance == 0.9

    # Verify count is still 1
    mems = get_active_user_memories(db, user_id="user_123")
    assert len(mems) == 1


def test_cross_user_isolation(db):
    """Verify User A's memories are NEVER returned to User B."""
    save_user_memory(db, user_id="user_A", memory_type="preference", memory_text="User loves black coffee")
    save_user_memory(db, user_id="user_B", memory_type="preference", memory_text="User loves masala chai")

    mems_a = get_active_user_memories(db, user_id="user_A")
    mems_b = get_active_user_memories(db, user_id="user_B")

    assert len(mems_a) == 1
    assert mems_a[0].memory_text == "User loves black coffee"
    assert len(mems_b) == 1
    assert mems_b[0].memory_text == "User loves masala chai"


def test_persona_relationship_memory_isolation(db):
    """Verify relationship memories for Luna are NOT available when talking to Ethan."""
    # Shared joke with Luna
    save_user_memory(
        db=db,
        user_id="user_A",
        profile_id="profile_1",
        memory_type="relationship",
        memory_text="Inside joke about cold brew spill",
    )
    # General preference (profile_id is None)
    save_user_memory(
        db=db,
        user_id="user_A",
        profile_id=None,
        memory_type="preference",
        memory_text="User loves jazz music",
    )

    # When chatting with Luna (profile_1): both should be visible
    luna_mems = get_active_user_memories(db, user_id="user_A", profile_id="profile_1")
    assert len(luna_mems) == 2

    # When chatting with Ethan (profile_2): only general preference should be visible
    ethan_mems = get_active_user_memories(db, user_id="user_A", profile_id="profile_2")
    assert len(ethan_mems) == 1
    assert ethan_mems[0].memory_text == "User loves jazz music"


def test_memory_soft_delete_and_forget(db):
    """Test soft deleting and bulk forgetting memories."""
    mem = save_user_memory(db, user_id="user_A", memory_type="preference", memory_text="Prefers evening texts")
    assert len(get_active_user_memories(db, user_id="user_A")) == 1

    # Delete single memory
    deleted = delete_user_memory(db, mem.id, user_id="user_A")
    assert deleted is True
    assert len(get_active_user_memories(db, user_id="user_A")) == 0

    # Test bulk forget
    save_user_memory(db, user_id="user_A", memory_type="profile_fact", memory_text="Lives in Delhi")
    save_user_memory(db, user_id="user_A", memory_type="preference", memory_text="Likes reading")
    assert len(get_active_user_memories(db, user_id="user_A")) == 2

    forgotten_count = forget_memories(db, user_id="user_A", scope="all")
    assert forgotten_count == 2
    assert len(get_active_user_memories(db, user_id="user_A")) == 0


def test_conversation_state_and_summary(db):
    """Test updating conversation state and rolling summaries."""
    # State test
    state = update_conversation_state(
        db=db,
        user_id="user_A",
        profile_id="profile_1",
        current_topic="photography exhibition",
        user_stated_emotion="excited",
        open_threads=["gallery opening"],
    )
    assert state.current_topic == "photography exhibition"
    assert state.user_stated_emotion == "excited"

    # Summary test
    s1 = save_summary(db, "user_A", "profile_1", "User and Luna discussed 35mm cameras.")
    assert s1.version == 1
    s2 = save_summary(db, "user_A", "profile_1", "User and Luna planned a photo walk in Brooklyn.")
    assert s2.version == 2

    latest = get_latest_summary(db, "user_A", "profile_1")
    assert latest.version == 2
    assert "photo walk" in latest.summary_text
