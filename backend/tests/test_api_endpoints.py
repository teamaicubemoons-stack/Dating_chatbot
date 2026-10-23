"""
tests/test_api_endpoints.py
---------------------------
Integration tests for FastAPI endpoints:
  - GET /profiles, GET /profiles/{id}
  - GET /memories, PATCH /memories/{id}, DELETE /memories/{id}, POST /memories/forget
  - GET /privacy/data-export, DELETE /account/data
  - GET /health
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from sqlalchemy.pool import StaticPool
from app.main import create_app
from app.models.db_models import Base, LLMStatus
from app.api.dependencies import get_db

# Create clean testing database with StaticPool so all threads share the same in-memory DB
TEST_DB_URL = "sqlite:///:memory:"
engine = create_engine(
    TEST_DB_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

# Seed LLM status
_db = TestingSessionLocal()
if _db.get(LLMStatus, 1) is None:
    _db.add(LLMStatus(id=1, current_provider="groq", groq_available=True))
    _db.commit()
_db.close()


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app = create_app()
app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)


def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "current_provider" in data


def test_profiles_endpoints():
    res = client.get("/profiles")
    assert res.status_code == 200
    profiles = res.json()
    assert isinstance(profiles, list)
    assert len(profiles) >= 1

    profile_id = profiles[0]["id"]
    single_res = client.get(f"/profiles/{profile_id}")
    assert single_res.status_code == 200
    assert single_res.json()["id"] == profile_id


def test_memories_crud_and_privacy_endpoints():
    user_id = "test_user_api_1"

    # Pre-populate a memory directly in test db
    from app.memory.memory_store import save_user_memory
    db = TestingSessionLocal()
    mem = save_user_memory(
        db=db,
        user_id=user_id,
        memory_type="preference",
        memory_text="User loves espresso",
        importance=0.8,
    )
    mem_id = mem.id
    db.close()

    # 1. GET /memories
    res = client.get(f"/memories?user_id={user_id}")
    assert res.status_code == 200
    mems = res.json()
    assert len(mems) == 1
    assert mems[0]["memory_text"] == "User loves espresso"

    # 2. PATCH /memories/{memory_id}
    patch_res = client.patch(
        f"/memories/{mem_id}?user_id={user_id}",
        json={"memory_text": "User loves double espresso"},
    )
    assert patch_res.status_code == 200
    assert patch_res.json()["memory_text"] == "User loves double espresso"

    # 3. GET /privacy/data-export
    export_res = client.get(f"/privacy/data-export?user_id={user_id}")
    assert export_res.status_code == 200
    export_data = export_res.json()
    assert export_data["user_id"] == user_id
    assert len(export_data["memories"]) == 1

    # 4. DELETE /memories/{memory_id}
    del_res = client.delete(f"/memories/{mem_id}?user_id={user_id}")
    assert del_res.status_code == 200

    # Verify empty after deletion
    res_after = client.get(f"/memories?user_id={user_id}")
    assert res_after.status_code == 200
    assert len(res_after.json()) == 0

    # 5. DELETE /account/data
    acc_del_res = client.delete(f"/account/data?user_id={user_id}")
    assert acc_del_res.status_code == 200
    assert acc_del_res.json()["success"] is True


def test_clear_chat_conversation_endpoint():
    from app.models.db_models import Message, Profile
    user_id = "test_user_chat_clear"
    profile_id = "diya_2b8989"

    db = TestingSessionLocal()
    # Ensure profile exists
    prof = db.query(Profile).filter(Profile.id == profile_id).first()
    if not prof:
        prof = Profile(id=profile_id, name="Diya", short_bio="Creative Soul")
        db.add(prof)

    # Insert test messages
    m1 = Message(user_id=user_id, profile_id=profile_id, role="user", content="Hello Diya")
    m2 = Message(user_id=user_id, profile_id=profile_id, role="assistant", content="Hi! How's your day?")
    db.add_all([m1, m2])
    db.commit()

    # Verify messages exist before deletion
    msgs_before = db.query(Message).filter(Message.user_id == user_id, Message.profile_id == profile_id).count()
    assert msgs_before == 2
    db.close()

    # Call DELETE /chat/{profile_id}/messages
    del_res = client.delete(f"/chat/{profile_id}/messages?user_id={user_id}")
    assert del_res.status_code == 200
    data = del_res.json()
    assert data["success"] is True
    assert data["deleted_messages"] == 2
    assert data["profile_id"] == profile_id

    # Verify in DB: messages are gone, but profile still exists!
    db = TestingSessionLocal()
    msgs_after = db.query(Message).filter(Message.user_id == user_id, Message.profile_id == profile_id).count()
    assert msgs_after == 0

    profile_still_exists = db.query(Profile).filter(Profile.id == profile_id).first()
    assert profile_still_exists is not None
    assert profile_still_exists.name == "Diya"
    db.close()


def test_nudge_user_endpoint(monkeypatch):
    from app.models.db_models import Message, Profile

    user_id = "test_user_nudge_1"
    profile_id = "profile_1"

    # 1. No messages yet -> nudged is False
    res_empty = client.post(f"/chat/{profile_id}/nudge", json={"user_id": user_id, "min_idle_seconds": 0})
    assert res_empty.status_code == 200
    assert res_empty.json()["nudged"] is False
    assert res_empty.json()["reason"] == "no_messages"

    # 2. User ended conversation with 'bye' -> nudged is False
    db = TestingSessionLocal()
    m_bye = Message(user_id=user_id, profile_id=profile_id, role="user", content="chalo bye baad me baat karenge")
    db.add(m_bye)
    db.commit()
    db.close()

    res_bye = client.post(f"/chat/{profile_id}/nudge", json={"user_id": user_id, "min_idle_seconds": 0})
    assert res_bye.status_code == 200
    assert res_bye.json()["nudged"] is False
    assert res_bye.json()["reason"] == "user_ended_conversation"

    # 3. User left mid-chat without saying bye
    user_id_active = "test_user_nudge_active"
    db = TestingSessionLocal()
    m_active = Message(user_id=user_id_active, profile_id=profile_id, role="user", content="horror movie ka scene hai")
    db.add(m_active)
    db.commit()
    db.close()

    # Mock route_and_generate to return a casual nudge
    monkeypatch.setattr(
        "app.chat.chat_engine.route_and_generate",
        lambda **kwargs: "kahan gayab ho gaye achanak se? haha 😂",
    )

    res_nudge = client.post(f"/chat/{profile_id}/nudge", json={"user_id": user_id_active, "min_idle_seconds": 0})
    assert res_nudge.status_code == 200
    data_nudge = res_nudge.json()
    assert data_nudge["nudged"] is True
    assert "kahan gayab" in data_nudge["reply"].lower()

    # 4. Immediate second nudge attempt -> blocked as already_nudged!
    res_dup = client.post(f"/chat/{profile_id}/nudge", json={"user_id": user_id_active, "min_idle_seconds": 0})
    assert res_dup.status_code == 200
    assert res_dup.json()["nudged"] is False
    assert res_dup.json()["reason"] == "already_nudged"


