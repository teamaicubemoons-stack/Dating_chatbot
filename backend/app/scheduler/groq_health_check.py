"""
scheduler/groq_health_check.py
-------------------------------
APScheduler background job that periodically probes Groq API availability.

Schedule: every 12 hours (configurable via settings.groq_health_check_interval_hours).
Behavior:
  - ONLY runs a check if current_provider == "claude" (no-op otherwise).
  - If Groq responds successfully → switches DB back to "groq".
  - If Groq still fails           → keeps "claude", updates last_checked_at timestamp.

This allows the system to self-heal automatically when Groq's daily quota resets,
without any manual intervention or visible disruption to users.
"""

import logging
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from app.api.dependencies import SessionLocal
from app.llm.llm_router import run_groq_health_check
from app.config.settings import settings

logger = logging.getLogger(__name__)

# Module-level scheduler singleton
_scheduler: BackgroundScheduler | None = None


def _health_check_job() -> None:
    """
    The actual job function executed by APScheduler.
    Opens its own DB session (not a request-scoped session) and delegates
    all logic to llm_router.run_groq_health_check().
    """
    logger.info("Running scheduled Groq health check...")
    db = SessionLocal()
    try:
        run_groq_health_check(db)
    except Exception as exc:
        logger.error("Groq health check job failed unexpectedly: %s", exc)
    finally:
        db.close()


def start_scheduler() -> None:
    """
    Start the APScheduler background scheduler with the Groq health check job.
    Called once at app startup. Safe to call multiple times (guards against re-init).
    """
    global _scheduler

    if _scheduler is not None and _scheduler.running:
        logger.debug("Scheduler already running — skipping re-init")
        return

    _scheduler = BackgroundScheduler(daemon=True)

    interval_hours = settings.groq_health_check_interval_hours
    _scheduler.add_job(
        func=_health_check_job,
        trigger=IntervalTrigger(hours=interval_hours),
        id="groq_health_check",
        name="Groq API Health Check",
        replace_existing=True,
        max_instances=1,  # prevent overlap if a check takes too long
    )

    _scheduler.start()
    logger.info(
        "Scheduler started — Groq health check every %d hour(s)", interval_hours
    )


def stop_scheduler() -> None:
    """
    Gracefully shut down the scheduler. Called in the FastAPI shutdown event.
    """
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped")
