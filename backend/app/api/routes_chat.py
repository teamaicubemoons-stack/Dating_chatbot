"""
api/routes_chat.py
------------------
FastAPI router for chat endpoints.

Endpoints:
  POST /chat/{profile_id}  → send a message, receive AI reply
  GET  /chat/{profile_id}/history → retrieve conversation history for a user

The chat endpoint is the primary hot path of the application.
All heavy lifting is delegated to chat_engine.py — this layer only handles
HTTP concerns: request parsing, response shaping, error mapping.
"""

import logging
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from app.chat.chat_engine import process_chat_message
from app.memory.memory_store import get_full_history
from app.models.schemas import ChatRequest, ChatResponse, ConversationHistoryResponse, MessageOut
from app.utils.error_handler import PersonaNotFoundError, LLMClientError
from app.api.dependencies import get_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/{profile_id}", response_model=ChatResponse)
async def send_message(
    profile_id: str,
    body: ChatRequest,
    db: Session = Depends(get_db),
):
    """
    Send a message to an AI persona and receive a reply.

    The backend handles all routing, RAG, memory, and LLM selection
    transparently — the response always contains just the reply string.
    """
    try:
        reply = await process_chat_message(
            db=db,
            profile_id=profile_id,
            user_id=body.user_id,
            user_message=body.message,
        )
        return ChatResponse(reply=reply, profile_id=profile_id)

    except PersonaNotFoundError:
        raise HTTPException(status_code=404, detail=f"Profile '{profile_id}' not found")

    except LLMClientError as exc:
        # Both providers failed — extremely rare, surface a generic error
        logger.error("Both LLM providers failed: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="The chat service is temporarily unavailable. Please try again shortly.",
        )

    except Exception as exc:
        logger.exception("Unexpected error in chat endpoint: %s", exc)
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")


@router.get("/{profile_id}/history", response_model=ConversationHistoryResponse)
def get_history(
    profile_id: str,
    user_id: str,
    db: Session = Depends(get_db),
):
    """
    Retrieve full conversation history for a user+profile pair.
    Used by the frontend to restore chat state when reopening a conversation.
    """
    messages = get_full_history(db, user_id, profile_id)
    return ConversationHistoryResponse(
        profile_id=profile_id,
        user_id=user_id,
        messages=[
            MessageOut(
                id=m.id,
                role=m.role,
                content=m.content,
                created_at=m.created_at,
            )
            for m in messages
        ],
    )


@router.get("/{profile_id}/opener")
def get_opener(profile_id: str):
    """
    Get a natural, random opening greeting for a persona when starting a fresh chat.
    Follows humanization rules: max 8-12 words, zero fake anecdotes, simple & friendly.
    """
    import random
    from app.personas.persona_loader import load_persona

    try:
        persona = load_persona(profile_id)
        openers = persona.get("initial_openers", [
            "Hey 🙂 How's your day going?",
            "Hey. What are you up to today?",
            "Hey. Glad you're here.",
            "Hey 😊 Anything interesting happen today?"
        ])
        opener = random.choice(openers)
        return {
            "profile_id": profile_id,
            "name": persona.get("name", "Companion"),
            "opener": opener
        }
    except PersonaNotFoundError:
        raise HTTPException(status_code=404, detail=f"Profile '{profile_id}' not found")
