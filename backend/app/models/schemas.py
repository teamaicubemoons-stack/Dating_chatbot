"""
models/schemas.py
-----------------
Pydantic v2 request/response schemas used by the FastAPI layer.
These are completely separate from SQLAlchemy models — one side handles
DB persistence, the other handles API serialization/validation.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Profile schemas
# ---------------------------------------------------------------------------

class ProfileSummary(BaseModel):
    """Lightweight profile info returned by GET /profiles list endpoint."""
    id: str
    name: str
    short_bio: str
    avatar_url: str
    country: Optional[str] = ""
    city: Optional[str] = ""
    age: Optional[int] = None
    gender: Optional[str] = ""

    model_config = {"from_attributes": True}


class ProfileDetail(BaseModel):
    """
    Full public profile info returned by GET /profiles/{profile_id}.
    NOTE: system_prompt_template and internal traits are intentionally
    excluded — never expose the raw system prompt to the frontend.
    """
    id: str
    name: str
    age: Optional[int] = None 
    gender: Optional[str] = ""
    country: Optional[str] = ""
    city: Optional[str] = ""
    short_bio: str
    avatar_url: str

    personality_traits: List[str] = []
    tone: str = ""
    relationship_intent: Optional[str] = ""

    model_config = {"from_attributes": True}


class ProfileCreate(BaseModel):
    """Request body for creating a new persona profile via POST /profiles."""
    name: str = Field(..., min_length=2, max_length=50)
    age: Optional[int] = Field(24, ge=18, le=99)
    gender: Optional[str] = Field("Female", description="Female, Male, or Non-binary")
    country: str = Field(..., min_length=2, max_length=60, description="Country of origin/residence")
    city: Optional[str] = Field("", max_length=60)
    short_bio: str = Field(..., min_length=5, max_length=200, description="Headline or one-liner for profile card")
    backstory: Optional[str] = Field("", max_length=3000, description="Detailed story, passions, quirks")
    tone: Optional[str] = Field("warm & playful", max_length=100)
    interests: Optional[List[str]] = Field(default_factory=list)
    personality_traits: Optional[List[str]] = Field(default_factory=list)
    relationship_intent: Optional[str] = Field("Looking for fun conversations & genuine connection", max_length=200)
    avatar_url: Optional[str] = Field("", description="Image URL or avatar path")
    initial_opener: Optional[str] = Field("", description="Custom greeting opener")
    humor_style: Optional[str] = Field("", max_length=160)
    playfulness: float = Field(0.5, ge=0.0, le=1.0)
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    emotional_expressiveness: float = Field(0.5, ge=0.0, le=1.0)
    flirting_style: Optional[str] = Field("warm and respectful", max_length=160)
    flirting_intensity: int = Field(1, ge=0, le=4)
    preferred_topics: List[str] = Field(default_factory=list, max_length=12)
    dislikes_and_boundaries: List[str] = Field(default_factory=list, max_length=12)
    language_preferences: List[str] = Field(default_factory=list, max_length=8)
    emoji_frequency: str = Field("occasional", pattern="^(never|rare|occasional|frequent)$")
    response_length: str = Field("short, conversational", max_length=80)

    @field_validator("preferred_topics", "personality_traits", "interests", "dislikes_and_boundaries", "language_preferences")
    @classmethod
    def validate_configured_lists(cls, values):
        for value in values or []:
            if not isinstance(value, str) or not value.strip() or len(value.strip()) > 100:
                raise ValueError("configuration entries must be non-empty strings of at most 100 characters")
        return [value.strip() for value in (values or [])]


# ---------------------------------------------------------------------------
# Chat schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """Body for POST /chat/{profile_id}."""
    user_id: str = Field(..., description="Anonymous session UUID or registered user ID")
    message: str = Field(..., min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    """Response returned after AI processes a chat message."""
    reply: str
    profile_id: str


class NudgeRequest(BaseModel):
    """Body for POST /chat/{profile_id}/nudge."""
    user_id: str = Field(..., description="User ID to check inactivity and send nudge for")
    min_idle_seconds: Optional[int] = Field(180, description="Minimum seconds of inactivity required to trigger nudge")


class NudgeResponse(BaseModel):
    """Response returned by inactivity check / nudge endpoint."""
    profile_id: str
    reply: Optional[str] = None
    nudged: bool = False
    reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Message history schemas
# ---------------------------------------------------------------------------

class MessageOut(BaseModel):
    id: int
    role: str          # "user" | "assistant"
    content: str
    created_at: datetime

    model_config = {"from_attributes": True}


class ConversationHistoryResponse(BaseModel):
    profile_id: str
    user_id: str
    messages: List[MessageOut]


# ---------------------------------------------------------------------------
# Memory Schemas
# ---------------------------------------------------------------------------

class UserMemoryOut(BaseModel):
    id: str
    user_id: str
    profile_id: Optional[str] = None
    memory_type: str
    memory_text: str
    normalized_key: Optional[str] = None
    confidence: float
    importance: float
    source_type: str
    valid_from: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    last_confirmed_at: Optional[datetime] = None
    status: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class UserMemoryPatch(BaseModel):
    memory_text: Optional[str] = Field(None, min_length=2, max_length=1000)
    importance: Optional[float] = Field(None, ge=0.0, le=1.0)
    status: Optional[str] = Field(None, pattern="^(active|archived|deleted)$")


class MemoryForgetRequest(BaseModel):
    user_id: str
    profile_id: Optional[str] = None
    memory_type: Optional[str] = None
    scope: str = Field("all", pattern="^(all|relationship|profile_facts)$")


class ConversationSummaryOut(BaseModel):
    id: int
    user_id: str
    profile_id: str
    summary_text: str
    version: int
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ConversationStateOut(BaseModel):
    user_id: str
    profile_id: str
    current_topic: str
    conversation_tone: str
    user_stated_emotion: str
    inferred_emotional_cue: str
    open_threads: List[str]
    last_follow_up_at: Optional[datetime]
    updated_at: datetime


class DataExportResponse(BaseModel):
    user_id: str
    exported_at: datetime
    memories: List[UserMemoryOut]
    summaries: List[ConversationSummaryOut]
    messages: List[MessageOut]
    states: List[Dict[str, Any]]


class AccountDeleteResponse(BaseModel):
    success: bool
    user_id: str
    deleted_records: Dict[str, int]
    message: str


# ---------------------------------------------------------------------------
# Health / status schemas
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    current_provider: str
    last_checked_at: Optional[datetime]
    groq_available: bool
