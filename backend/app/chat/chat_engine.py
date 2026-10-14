"""
chat/chat_engine.py
--------------------
The central orchestrator of the AI chat pipeline.

Flow:
  1. Load persona configuration.
  2. Retrieve RAG grounding facts & scored long-term user memories.
  3. Fetch rolling summary & active conversation state.
  4. Fetch recent conversation history turns.
  5. Assemble modular prompt with human texting guidelines & language matching.
  6. Generate AI response via resilient LLM router (Groq -> Claude fallback).
  7. Validate and sanitize generated response (no metadata leaks, no consecutive repetitions).
  8. Persist conversation turn in SQLite messages table.
  9. Run asynchronous/safe long-term memory extraction and state update.
"""

import logging
import json
from datetime import datetime
from typing import Any, Optional
from sqlalchemy.orm import Session

from app.personas.persona_loader import load_persona
from app.rag.retriever import retrieve_context
from app.memory.conversation_memory import get_conversation_history, persist_exchange
from app.memory.memory_store import (
    ensure_user, ensure_profile, get_latest_summary, get_or_create_conversation_state
)
from app.memory.memory_extractor import (
    process_and_persist_memories, detect_pronoun_preference, detect_user_gender_cue
)
from app.utils.time_context import get_current_context
from app.chat.prompt_builder import build_final_prompt
from app.chat.response_validator import validate_and_clean_response
from app.llm.llm_router import route_and_generate
from app.utils.error_handler import PersonaNotFoundError, LLMClientError

logger = logging.getLogger(__name__)


async def process_chat_message(
    db: Session,
    profile_id: str,
    user_id: str,
    user_message: str,
    background_tasks: Any = None,
) -> str:
    """
    Full pipeline: receive a user message and return the AI persona's reply.
    """
    logger.info("Chat request — profile: %s | user: %s", profile_id, user_id[:8])

    # 1. Load persona config
    persona = load_persona(profile_id)
    persona_name = persona.get("name", "Companion")

    # 2. Ensure DB records exist for this user+profile
    ensure_user(db, user_id)
    ensure_profile(db, profile_id, persona_name, persona.get("short_bio", ""))

    # 3. Retrieve RAG grounding context & scored long-term user memories
    rag_context = retrieve_context(
        profile_id=profile_id,
        user_message=user_message,
        user_id=user_id,
        db=db,
    )

    # 4. Fetch rolling conversation summary & active conversation state
    summary_record = get_latest_summary(db, user_id, profile_id)
    rolling_summary = summary_record.summary_text if summary_record else None

    # Authoritative runtime date/time context
    time_context = get_current_context()

    state_record = get_or_create_conversation_state(db, user_id, profile_id)
    recent_ctx = {}
    try:
        recent_ctx = json.loads(state_record.recent_context_json or "{}")
    except Exception:
        recent_ctx = {}

    # Extract pronoun preference (default "tum") & user grammatical gender
    preferred_pronoun = recent_ctx.get("preferred_pronoun", "tum")
    user_gender = recent_ctx.get("user_gender")

    # Dynamic detection from current message
    msg_pronoun = detect_pronoun_preference(user_message)
    if msg_pronoun:
        preferred_pronoun = msg_pronoun
        recent_ctx["preferred_pronoun"] = msg_pronoun

    msg_gender = detect_user_gender_cue(user_message)
    if msg_gender:
        user_gender = msg_gender
        recent_ctx["user_gender"] = msg_gender

    # 5. Fetch recent conversation history turns
    chat_history = get_conversation_history(db, user_id, profile_id)

    # If gender wasn't in state or current message, inspect recent user turns
    if not user_gender and chat_history:
        for turn in reversed(chat_history):
            if turn.get("role") == "user":
                hist_gender = detect_user_gender_cue(turn.get("content", ""))
                if hist_gender:
                    user_gender = hist_gender
                    recent_ctx["user_gender"] = hist_gender
                    break

    # Session staleness check: If last session was > 4 hours ago, do not blindly carry forward old topics
    is_stale_session = False
    if state_record.updated_at:
        delta_sec = (datetime.utcnow() - state_record.updated_at).total_seconds()
        if delta_sec > 14400:
            is_stale_session = True

    state_dict = {
        "current_topic": None if is_stale_session else state_record.current_topic,
        "user_stated_emotion": state_record.user_stated_emotion,
        "open_threads": [] if is_stale_session else json.loads(state_record.open_threads_json or "[]"),
        "active_style_feedback": recent_ctx.get("active_style_feedback"),
        "preferred_pronoun": preferred_pronoun,
        "user_gender": user_gender,
    }

    # 6. Build modular final prompt with authoritative runtime time context
    system_prompt, trimmed_history = build_final_prompt(
        persona=persona,
        rag_context=rag_context,
        chat_history=chat_history,
        user_message=user_message,
        rolling_summary=rolling_summary,
        conversation_state=state_dict,
        time_context=time_context,
    )

    # 7. Generate AI response (transparent Groq/Claude failover routing)
    raw_reply = route_and_generate(
        db=db,
        system_prompt=system_prompt,
        chat_history=trimmed_history,
        user_message=user_message,
    )

    # 8. Deterministic response validation & cleanup
    ai_reply = validate_and_clean_response(
        raw_response=raw_reply,
        persona_name=persona_name,
        recent_history=trimmed_history,
        user_gender=user_gender,
        preferred_pronoun=preferred_pronoun,
        user_message=user_message,
    )

    # 9. Persist conversation turn in SQLite
    user_msg_record, asst_msg_record = persist_exchange(
        db=db,
        user_id=user_id,
        profile_id=profile_id,
        user_message=user_message,
        ai_reply=ai_reply,
    )

    # 10. Extract, deduplicate, and persist long-term candidate memories
    # Use non-blocking background task so HTTP response returns immediately
    source_msg_id = user_msg_record.id if user_msg_record else None
    if background_tasks:
        def _bg_worker(u_id: str, p_id: str, msg: str, s_id: Any):
            from app.api.dependencies import SessionLocal
            bg_db = SessionLocal()
            try:
                process_and_persist_memories(
                    db=bg_db,
                    user_id=u_id,
                    profile_id=p_id,
                    user_message=msg,
                    source_message_id=s_id,
                )
            except Exception as bg_exc:
                logger.warning("Background memory extraction error: %s", bg_exc)
            finally:
                bg_db.close()

        background_tasks.add_task(_bg_worker, user_id, profile_id, user_message, source_msg_id)
        logger.debug("Scheduled background memory extraction for user %s", user_id[:8])
    else:
        # Fallback to in-process execution (e.g. in test suites)
        process_and_persist_memories(
            db=db,
            user_id=user_id,
            profile_id=profile_id,
            user_message=user_message,
            source_message_id=source_msg_id,
        )

    logger.info("Reply generated & persisted (%d chars) for profile %s", len(ai_reply), profile_id)
    return ai_reply
