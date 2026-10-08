"""
api/routes_health.py
---------------------
Internal health-check endpoint.

GET /health → returns current LLM provider status, DB connectivity.

This endpoint is for INTERNAL MONITORING ONLY.
It is intentionally NOT linked from the frontend UI in any user-facing way.
Never expose provider names or internal state in chat responses.
"""

import logging
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.llm.llm_router import get_current_provider
from app.models.db_models import LLMStatus
from app.models.schemas import HealthResponse
from app.api.dependencies import get_db

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check(db: Session = Depends(get_db)):
    """
    Returns current system health including active LLM provider.
    Intended for internal monitoring dashboards — not shown in the UI.
    """
    try:
        status = db.get(LLMStatus, 1)
        current_provider = status.current_provider if status else "groq"
        last_checked = status.last_checked_at if status else None
        groq_available = status.groq_available if status else True

        return HealthResponse(
            status="ok",
            current_provider=current_provider,
            last_checked_at=last_checked,
            groq_available=groq_available,
        )
    except Exception as exc:
        logger.error("Health check DB error: %s", exc)
        return HealthResponse(
            status="degraded",
            current_provider="unknown",
            last_checked_at=None,
            groq_available=False,
        )
