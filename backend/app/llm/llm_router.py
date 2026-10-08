"""
llm/llm_router.py
-----------------
LLM routing engine. THIS IS THE ONLY FILE that knows about both providers.

Routing rules (exact spec):
  1. Default provider at startup: "groq".
  2. On every chat call:
     - If current_provider == "groq"  → call GroqClient.
       • On GroqQuotaExceededError   → switch DB to "claude", retry with ClaudeClient.
         The user receives a seamless reply — no error, no visible delay added.
     - If current_provider == "claude" → call ClaudeClient directly (skip Groq test).
  3. Background health-check job (APScheduler, every 12h) probes Groq:
     - Runs ONLY when current_provider == "claude".
     - If Groq responds OK → flip DB back to "groq".
     - If still failing   → keep "claude", update last_checked_at.
  4. Nothing in this file leaks provider names to the API or chat layers.

The DB-backed LLMStatus row (id=1) is the single source of truth for active provider.
All DB writes use a short-lived session to prevent connection leaks.
"""

import logging
from datetime import datetime
from typing import List, Dict

from sqlalchemy.orm import Session

from app.llm.groq_client import GroqClient
from app.llm.claude_client import ClaudeClient
from app.models.db_models import LLMStatus
from app.utils.error_handler import GroqQuotaExceededError, LLMClientError

logger = logging.getLogger(__name__)

# Lazy singletons — created on first use, not at import time.
# This prevents startup crashes when API keys are not yet configured.
_groq_client: GroqClient | None = None
_claude_client: ClaudeClient | None = None


def _get_groq() -> GroqClient:
    global _groq_client
    if _groq_client is None:
        _groq_client = GroqClient()
    return _groq_client


def _get_claude() -> ClaudeClient:
    global _claude_client
    if _claude_client is None:
        _claude_client = ClaudeClient()
    return _claude_client


def _get_or_create_status(db: Session) -> LLMStatus:
    """Fetch the canonical LLMStatus row, creating it (as Groq) if absent."""
    status = db.get(LLMStatus, 1)
    if status is None:
        status = LLMStatus(id=1, current_provider="groq", groq_available=True)
        db.add(status)
        db.commit()
        db.refresh(status)
    return status


def get_current_provider(db: Session) -> str:
    """Return the active provider name — useful for the /health endpoint."""
    return _get_or_create_status(db).current_provider


def _set_provider(db: Session, provider: str, groq_available: bool) -> None:
    """Persist a provider switch to DB."""
    status = _get_or_create_status(db)
    status.current_provider = provider
    status.groq_available = groq_available
    status.last_checked_at = datetime.utcnow()
    db.commit()
    logger.info("LLM provider switched → %s", provider)


def route_and_generate(
    db: Session,
    system_prompt: str,
    chat_history: List[Dict[str, str]],
    user_message: str,
) -> str:
    """
    Main entry point for all chat generation calls.

    Reads the active provider from DB, calls the appropriate client,
    handles fallback silently, and returns the AI reply string.

    Args:
        db:           SQLAlchemy session (caller-owned, caller closes).
        system_prompt: Fully constructed system prompt from prompt_builder.
        chat_history:  Prior turns as [{"role": ..., "content": ...}].
        user_message:  Current user input.

    Returns:
        AI-generated reply string.

    Raises:
        LLMClientError: if both providers fail (should be extremely rare).
    """
    status = _get_or_create_status(db)
    provider = status.current_provider

    if provider == "groq":
        try:
            return _get_groq().generate_response(system_prompt, chat_history, user_message)
        except GroqQuotaExceededError:
            # Quota hit — silently switch to Claude and retry the same request
            logger.warning("Groq quota exceeded; switching to Claude (transparent to user)")
            _set_provider(db, "claude", groq_available=False)
            # Fall through to Claude call below
            return _get_claude().generate_response(system_prompt, chat_history, user_message)
        except LLMClientError:
            # Non-quota Groq error — still try Claude as emergency fallback
            logger.error("Groq non-quota error; attempting Claude emergency fallback")
            return _get_claude().generate_response(system_prompt, chat_history, user_message)

    else:  # provider == "claude"
        return _get_claude().generate_response(system_prompt, chat_history, user_message)


def run_groq_health_check(db: Session) -> None:
    """
    Called by APScheduler every 12 hours.
    Only probes Groq when current_provider == "claude" (no-op otherwise).
    Silently updates DB based on result.
    """
    status = _get_or_create_status(db)
    if status.current_provider != "claude":
        logger.debug("Health check skipped — Groq is already active provider")
        return

    logger.info("Running Groq health check probe...")
    groq_ok = _get_groq().health_check()

    if groq_ok:
        _set_provider(db, "groq", groq_available=True)
        logger.info("Groq restored — switched back to Groq as primary provider")
    else:
        # Just update the timestamp, keep Claude
        status.last_checked_at = datetime.utcnow()
        db.commit()
        logger.info("Groq still unavailable — remaining on Claude")
