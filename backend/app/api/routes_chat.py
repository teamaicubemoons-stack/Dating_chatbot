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

import asyncio
import logging
from datetime import timezone
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from sqlalchemy.orm import Session

from app.chat.chat_engine import process_chat_message, process_nudge_message
from app.chat.typing_delay import calculate_typing_delay
from app.config.settings import settings
from app.memory.memory_store import get_full_history
from app.models.schemas import (
    ChatRequest, ChatResponse, ConversationHistoryResponse, MessageOut,
    NudgeRequest, NudgeResponse,
)
from app.utils.error_handler import PersonaNotFoundError, LLMClientError
from app.api.dependencies import get_db
from fastapi.responses import FileResponse, HTMLResponse
from pathlib import Path

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


@router.get("/{profile_id}", include_in_schema=False)
@router.get("/{profile_id}/", include_in_schema=False)
def serve_chat_page(profile_id: str):
    """
    Handle GET /chat/{profile_id} for browser page refreshes and direct links.
    Serves the React frontend SPA so React Router can render the ChatPage.
    """
    base_dir = Path(__file__).resolve().parent.parent.parent
    possible_dist_paths = [
        base_dir.parent / "frontend" / "dist",
        base_dir / "static",
        base_dir / "frontend" / "dist",
        base_dir.parent / "static",
        Path("/app/frontend/dist"),
        Path("/home/user/app/frontend/dist"),
    ]
    for dist in possible_dist_paths:
        idx = dist / "index.html"
        if idx.is_file():
            return FileResponse(idx)
    return HTMLResponse("<h1>Spark Companion — Loading...</h1>")


@router.post("/{profile_id}", response_model=ChatResponse)
async def send_message(
    profile_id: str,
    body: ChatRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Send a message to an AI persona and receive a reply.

    The backend handles all routing, RAG, memory, and LLM selection
    transparently — the response always contains just the reply string.
    Applies a natural, length-calibrated typing delay before returning.
    """
    try:
        reply = await process_chat_message(
            db=db,
            profile_id=profile_id,
            user_id=body.user_id,
            user_message=body.message,
            background_tasks=background_tasks,
        )

        # Apply natural, length-proportional human typing delay
        if getattr(settings, "enable_typing_delay", True):
            delay = calculate_typing_delay(reply)
            if delay > 0:
                logger.info(
                    "Pacing response for profile %s: sleeping %.2fs (length: %d chars)",
                    profile_id,
                    delay,
                    len(reply),
                )
                await asyncio.sleep(delay)

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


@router.post("/{profile_id}/nudge", response_model=NudgeResponse)
async def nudge_user(
    profile_id: str,
    body: NudgeRequest,
    db: Session = Depends(get_db),
):
    """
    Check if the user has been idle for 3-4 minutes without ending the conversation.
    If eligible, generates and persists exactly ONE proactive check-in message.
    """
    try:
        min_idle = body.min_idle_seconds if body.min_idle_seconds is not None else 180
        reply, reason = await process_nudge_message(
            db=db,
            profile_id=profile_id,
            user_id=body.user_id,
            min_idle_seconds=min_idle,
        )
        if reply:
            return NudgeResponse(
                profile_id=profile_id,
                reply=reply,
                nudged=True,
                reason="success",
            )
        return NudgeResponse(
            profile_id=profile_id,
            reply=None,
            nudged=False,
            reason=reason,
        )
    except PersonaNotFoundError:
        raise HTTPException(status_code=404, detail=f"Profile '{profile_id}' not found")
    except Exception as exc:
        logger.exception("Error in nudge endpoint: %s", exc)
        return NudgeResponse(
            profile_id=profile_id,
            reply=None,
            nudged=False,
            reason="error",
        )


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
                created_at=(
                    m.created_at.replace(tzinfo=timezone.utc)
                    if m.created_at and m.created_at.tzinfo is None
                    else m.created_at
                ),
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


@router.delete("/{profile_id}/messages")
def clear_chat_conversation(
    profile_id: str,
    user_id: str,
    db: Session = Depends(get_db),
):
    """
    Delete all chat messages between user and a specific persona profile.
    Resets the conversation to a fresh state without deleting the profile itself.
    """
    from app.models.db_models import Message, ConversationSummary, ConversationState

    deleted_count = (
        db.query(Message)
        .filter(Message.user_id == user_id, Message.profile_id == profile_id)
        .delete()
    )

    # Clear rolling summary and conversation state for this specific chat
    db.query(ConversationSummary).filter(
        ConversationSummary.user_id == user_id,
        ConversationSummary.profile_id == profile_id,
    ).delete()

    db.query(ConversationState).filter(
        ConversationState.user_id == user_id,
        ConversationState.profile_id == profile_id,
    ).delete()

    db.commit()
    logger.info("Cleared %d messages for user %s and profile %s", deleted_count, user_id[:8], profile_id)
    return {
        "success": True,
        "profile_id": profile_id,
        "deleted_messages": deleted_count,
        "message": f"Conversation with {profile_id} cleared successfully.",
    }

