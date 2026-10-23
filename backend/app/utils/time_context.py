"""
utils/time_context.py
---------------------
Provides authoritative runtime date, day, and time context.
Used to inject real-time clock data into system prompts so the LLM
never hallucinates or guesses dates, weekdays, or time periods.
"""

from datetime import datetime, date
from zoneinfo import ZoneInfo
from typing import Dict, Any


def get_current_context(timezone_str: str = "Asia/Kolkata") -> Dict[str, Any]:
    """
    Generate authoritative current date/time context for the specified timezone.
    Default is Indian Standard Time (IST / Asia/Kolkata).
    """
    try:
        tz = ZoneInfo(timezone_str)
    except Exception:
        tz = ZoneInfo("Asia/Kolkata")

    now = datetime.now(tz)

    hour = now.hour
    if 5 <= hour < 10:
        day_period = "morning"
    elif 10 <= hour < 12:
        day_period = "late morning"
    elif 12 <= hour < 17:
        day_period = "afternoon"
    elif 17 <= hour < 21:
        day_period = "evening"
    else:
        day_period = "night"

    context = {
        "current_datetime": now.isoformat(),
        "current_date": now.strftime("%Y-%m-%d"),
        "day_name": now.strftime("%A"),
        "local_time": now.strftime("%I:%M %p"),
        "timezone": timezone_str,
        "day_period": day_period,
    }
    context["verified_event_context"] = _get_verified_event_context(now.date())
    return context


def _get_verified_event_context(current_date: date) -> Dict[str, Any]:
    """Return conservative India event status for dates we explicitly verify."""
    # Keep this small and date-backed. Unknown regional events are omitted
    # rather than guessed, and callers can still use the user's own context.
    calendars = {
        2026: {
            "navratri": (date(2026, 10, 11), date(2026, 10, 19)),
            "dussehra": (date(2026, 10, 20), date(2026, 10, 20)),
            "dhanteras": (date(2026, 11, 6), date(2026, 11, 6)),
            "diwali": (date(2026, 11, 8), date(2026, 11, 8)),
        }
    }
    events = calendars.get(current_date.year, {})
    statuses = {}
    for name, (start, end) in events.items():
        if start <= current_date <= end:
            statuses[name] = {"status": "today", "date": start.isoformat()}
        elif current_date < start:
            statuses[name] = {"status": "upcoming", "date": start.isoformat()}
        else:
            statuses[name] = {"status": "past", "date": end.isoformat()}
    return {
        "region": "India",
        "source": "verified application calendar; regional observances may vary",
        "events": statuses,
    }
