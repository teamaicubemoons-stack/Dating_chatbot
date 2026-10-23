"""
scheduler/openai_health_check.py
---------------------------------
APScheduler background job that periodically probes OpenAI API availability.

Uses a free `models.list()` call — no chat completions, no token cost.
Schedule: every 6 hours (configurable via settings.openai_health_check_interval_hours).
"""

from __future__ import annotations

import logging
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from app.api.dependencies import SessionLocal
from app.llm.llm_router import run_health_check
from app.config.settings import settings

logger = logging.getLogger(__name__)

_scheduler: BackgroundScheduler | None = None


def _health_check_job() -> None:
    logger.info("Running scheduled OpenAI health check (models.list)...")
    db = SessionLocal()
    try:
        ok = run_health_check(db)
        if not ok:
            logger.warning("OpenAI health check failed — will retry next interval")
    except Exception as exc:
        logger.error("OpenAI health check job crashed: %s", exc)
    finally:
        db.close()


def start_scheduler() -> None:
    """Start APScheduler. Safe to call multiple times (guards against re-init)."""
    global _scheduler
    if _scheduler is not None and _scheduler.running:
        logger.debug("Scheduler already running — skipping re-init")
        return

    _scheduler = BackgroundScheduler(daemon=True)
    interval_hours = settings.openai_health_check_interval_hours
    _scheduler.add_job(
        func=_health_check_job,
        trigger=IntervalTrigger(hours=interval_hours),
        id="openai_health_check",
        name="OpenAI API Health Check",
        replace_existing=True,
        max_instances=1,
    )
    _scheduler.start()
    logger.info("Scheduler started — OpenAI health check every %d hour(s)", interval_hours)


def stop_scheduler() -> None:
    """Gracefully shut down the scheduler."""
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped")
