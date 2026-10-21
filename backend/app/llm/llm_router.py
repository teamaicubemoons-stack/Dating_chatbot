"""
llm/llm_router.py
-----------------
LLM routing engine.

NOTE: Groq failover code is kept intact but commented out below for testing OpenAI directly.
It can easily be re-enabled later by uncommenting the designated section.
"""

import logging
from datetime import datetime
from typing import List, Dict

from sqlalchemy.orm import Session

from app.llm.groq_client import GroqClient
from app.llm.openai_client import OpenAIClient
from app.llm.claude_client import ClaudeClient
from app.models.db_models import LLMStatus
from app.utils.error_handler import GroqQuotaExceededError, LLMClientError

logger = logging.getLogger(__name__)

# Lazy singletons — created on first use, not at import time.
_groq_client: GroqClient | None = None
_openai_client: OpenAIClient | None = None
_claude_client: ClaudeClient | None = None


def _get_groq() -> GroqClient:
    global _groq_client
    if _groq_client is None:
        _groq_client = GroqClient()
    return _groq_client


def _get_openai() -> OpenAIClient:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAIClient()
    return _openai_client


def _get_claude() -> ClaudeClient:
    global _claude_client
    if _claude_client is None:
        _claude_client = ClaudeClient()
    return _claude_client


def _get_or_create_status(db: Session) -> LLMStatus:
    """Fetch the canonical LLMStatus row, creating it (as OpenAI for testing) if absent."""
    status = db.get(LLMStatus, 1)
    if status is None:
        status = LLMStatus(id=1, current_provider="openai", groq_available=False)
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

    Currently configured for testing: Groq failover logic is commented out
    so OpenAI is used directly for all responses.
    """
    status = _get_or_create_status(db)
    import time
    t_start = time.perf_counter()

    # =========================================================================
    # [TEMPORARILY COMMENTED OUT FOR TESTING OPENAI DIRECTLY]
    # To re-enable Groq primary + OpenAI fallback later, uncomment this block:
    # -------------------------------------------------------------------------
    # provider = status.current_provider
    # if provider == "groq":
    #     try:
    #         res = _get_groq().generate_response(system_prompt, chat_history, user_message)
    #         logger.info("Routing: Groq served request in %.2fs", time.perf_counter() - t_start)
    #         return res
    #     except GroqQuotaExceededError:
    #         # Quota hit — silently switch to OpenAI and retry the same request
    #         logger.warning("Groq quota exceeded after %.2fs; switching to OpenAI (transparent to user)", time.perf_counter() - t_start)
    #         _set_provider(db, "openai", groq_available=False)
    #         try:
    #             res = _get_openai().generate_response(system_prompt, chat_history, user_message)
    #             logger.info("Routing: OpenAI fallback served request in %.2fs total", time.perf_counter() - t_start)
    #             return res
    #         except Exception as o_err:
    #             logger.warning("OpenAI fallback failed (%s); reverting DB to Groq", o_err)
    #             _set_provider(db, "groq", groq_available=True)
    #             raise
    #     except LLMClientError as err:
    #         # Non-quota Groq error — still try OpenAI as emergency fallback
    #         logger.error("Groq non-quota error after %.2fs (%s); attempting OpenAI emergency fallback", time.perf_counter() - t_start, err)
    #         try:
    #             res = _get_openai().generate_response(system_prompt, chat_history, user_message)
    #             logger.info("Routing: OpenAI emergency fallback served request in %.2fs total", time.perf_counter() - t_start)
    #             return res
    #         except Exception as o_err:
    #             logger.warning("OpenAI emergency fallback failed (%s)", o_err)
    #             raise
    #
    # else:  # provider in ["openai", "claude"]
    #     try:
    #         res = _get_openai().generate_response(system_prompt, chat_history, user_message)
    #         logger.info("Routing: OpenAI served request in %.2fs", time.perf_counter() - t_start)
    #         return res
    #     except Exception as o_err:
    #         logger.warning("OpenAI unavailable (%s); attempting Groq recovery", o_err)
    #         _set_provider(db, "groq", groq_available=True)
    #         res = _get_groq().generate_response(system_prompt, chat_history, user_message)
    #         logger.info("Routing: Groq recovery served request in %.2fs", time.perf_counter() - t_start)
    #         return res
    # =========================================================================

    # Directly serve all requests via OpenAI for testing:
    if status.current_provider != "openai":
        _set_provider(db, "openai", groq_available=False)

    res = _get_openai().generate_response(system_prompt, chat_history, user_message)
    logger.info("Routing: OpenAI direct served request in %.2fs", time.perf_counter() - t_start)
    return res


def run_groq_health_check(db: Session) -> None:
    """
    Called by APScheduler every 12 hours.
    (Temporarily paused while Groq is commented out for testing OpenAI directly).
    """
    logger.debug("Groq health check paused — testing OpenAI directly")
    return
