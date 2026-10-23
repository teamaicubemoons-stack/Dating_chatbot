"""
llm/llm_router.py
-----------------
Single-provider LLM routing: OpenAI only.

Groq and Claude are removed entirely (no dead imports, no commented code).
Primary: gpt-5.6-luna  |  Fallback (on failure): gpt-5.6-terra
Both are handled inside OpenAIClient; the router just delegates.
"""

from __future__ import annotations

import logging
import time
from typing import List, Dict, Optional

from sqlalchemy.orm import Session

from app.llm.openai_client import OpenAIClient
from app.models.db_models import LLMStatus
from app.utils.error_handler import LLMClientError

logger = logging.getLogger(__name__)

# Lazy singleton — created on first use, not at import time
_openai_client: OpenAIClient | None = None


def _get_openai() -> OpenAIClient:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAIClient()
    return _openai_client


def _get_or_create_status(db: Session) -> LLMStatus:
    """Fetch the canonical LLMStatus row (always 'openai' now)."""
    status = db.get(LLMStatus, 1)
    if status is None:
        status = LLMStatus(id=1, current_provider="openai", groq_available=False)
        db.add(status)
        db.commit()
        db.refresh(status)
    elif status.current_provider != "openai":
        status.current_provider = "openai"
        db.commit()
    return status


def get_current_provider(db: Session) -> str:
    return _get_or_create_status(db).current_provider


def route_and_generate(
    db: Session,
    system_prompt: str,
    chat_history: List[Dict[str, str]],
    user_message: str,
    user_id: Optional[str] = None,
) -> str:
    """
    Main entry point for all chat generation calls.
    Always uses OpenAI (primary: gpt-5.6-luna, fallback: gpt-5.6-terra).
    """
    _get_or_create_status(db)
    t_start = time.perf_counter()
    try:
        res = _get_openai().generate_response(
            system_prompt=system_prompt,
            chat_history=chat_history,
            user_message=user_message,
            user_id=user_id,
        )
        logger.info(
            "route_and_generate: served in %.2fs", time.perf_counter() - t_start
        )
        return res
    except LLMClientError:
        raise
    except Exception as exc:
        logger.error(
            "route_and_generate: unexpected error after %.2fs: %s",
            time.perf_counter() - t_start, exc,
        )
        raise LLMClientError(f"LLM routing failed: {exc}") from exc


def run_health_check(db: Session) -> bool:
    """
    Called by APScheduler periodically.
    Uses a free models.list() call — no chat completions, no token cost.
    """
    ok = _get_openai().health_check()
    status = _get_or_create_status(db)
    from datetime import datetime
    status.last_checked_at = datetime.utcnow()
    status.groq_available = False  # always False; field kept for schema compat
    db.commit()
    logger.info("OpenAI health check: %s", "OK" if ok else "FAILED")
    return ok


# ── backward-compat shim (scheduler imports this name) ──────────────────────
def run_groq_health_check(db: Session) -> None:
    """Deprecated alias — delegates to run_health_check."""
    run_health_check(db)
