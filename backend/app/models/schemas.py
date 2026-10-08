"""
models/schemas.py
-----------------
Pydantic v2 request/response schemas used by the FastAPI layer.
These are completely separate from SQLAlchemy models — one side handles
DB persistence, the other handles API serialization/validation.
"""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


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
# Health / status schemas
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    current_provider: str
    last_checked_at: Optional[datetime]
    groq_available: bool
