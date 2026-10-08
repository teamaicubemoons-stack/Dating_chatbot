"""
chat/chat_engine.py
--------------------
The central orchestrator of the AI chat pipeline.

This is the ONLY module that imports from all subsystems simultaneously.
It coordinates the exact flow specified in the architecture:

  1. Load persona config       → persona_loader.py
  2. Retrieve RAG context      → retriever.py
  3. Fetch recent history      → conversation_memory.py
  4. Build final prompt        → prompt_builder.py
  5. Generate AI response      → llm_router.py   (handles Groq↔Claude silently)
  6. Persist exchange          → conversation_memory.py + memory_store.py

chat_engine.py has ZERO knowledge of:
  - Which LLM provider is active (routing is fully internal to llm_router)
  - Where embeddings come from (abstracted by retriever)
  - How history is stored (abstracted by memory_store)

This isolation ensures that swapping any subsystem requires changes in only
that subsystem's module — not here.
"""

import logging
from sqlalchemy.orm import Session

from app.personas.persona_loader import load_persona
from app.rag.retriever import retrieve_context
from app.memory.conversation_memory import get_conversation_history, persist_exchange
from app.memory.memory_store import ensure_user, ensure_profile
from app.chat.prompt_builder import build_final_prompt
from app.llm.llm_router import route_and_generate
from app.utils.error_handler import PersonaNotFoundError, LLMClientError

logger = logging.getLogger(__name__)


async def process_chat_message(
    db: Session,
    profile_id: str,
    user_id: str,
    user_message: str,
) -> str:
    """
    Full pipeline: receive a user message and return the AI persona's reply.

    Args:
        db:           SQLAlchemy session (FastAPI dependency injection provides this).
        profile_id:   Which persona the user is chatting with.
        user_id:      The user's session/account ID.
        user_message: The text the user just sent.

    Returns:
        The AI persona's reply string.

    Raises:
        PersonaNotFoundError: if the profile_id doesn't exist.
        LLMClientError:       if all LLM providers fail (should be very rare).
    """
    logger.info("Chat request — profile: %s | user: %s", profile_id, user_id[:8])

    # --- Step 1: Load persona config (fast — reads from disk cache) ---
    persona = load_persona(profile_id)
    logger.debug("Persona loaded: %s", persona["name"])

    # --- Ensure DB records exist for this user+profile ---
    ensure_user(db, user_id)
    ensure_profile(db, profile_id, persona.get("name", ""), persona.get("short_bio", ""))

    # --- Step 2: Retrieve RAG context ---
    rag_context = retrieve_context(
        profile_id=profile_id,
        user_message=user_message,
        user_id=user_id,
    )
    logger.debug("RAG context: %d chars", len(rag_context))

    # --- Step 3: Fetch recent conversation history ---
    chat_history = get_conversation_history(db, user_id, profile_id)
    logger.debug("History: %d messages", len(chat_history))

    # --- Step 4: Build final prompt ---
    system_prompt, trimmed_history = build_final_prompt(
        persona=persona,
        rag_context=rag_context,
        chat_history=chat_history,
        user_message=user_message,
    )

    # --- Step 5: Generate AI response (LLM routing is fully transparent here) ---
    ai_reply = route_and_generate(
        db=db,
        system_prompt=system_prompt,
        chat_history=trimmed_history,
        user_message=user_message,
    )
    logger.info("Reply generated (%d chars) for profile %s", len(ai_reply), profile_id)

    # --- Step 6: Persist the exchange ---
    persist_exchange(
        db=db,
        user_id=user_id,
        profile_id=profile_id,
        user_message=user_message,
        ai_reply=ai_reply,
    )

    return ai_reply
