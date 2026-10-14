"""
utils/time_context.py
---------------------
Provides authoritative runtime date, day, and time context.
Used to inject real-time clock data into system prompts so the LLM
never hallucinates or guesses dates, weekdays, or time periods.
"""

from datetime import datetime
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
    if 5 <= hour < 12:
        day_period = "morning"
    elif 12 <= hour < 17:
        day_period = "afternoon"
    elif 17 <= hour < 21:
        day_period = "evening"
    else:
        day_period = "night"

    return {
        "current_datetime": now.isoformat(),
        "current_date": now.strftime("%Y-%m-%d"),
        "day_name": now.strftime("%A"),
        "local_time": now.strftime("%I:%M %p"),
        "timezone": timezone_str,
        "day_period": day_period,
    }
