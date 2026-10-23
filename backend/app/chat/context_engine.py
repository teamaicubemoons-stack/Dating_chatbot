"""
chat/context_engine.py
-----------------------
Builds the [LIVE CONTEXT] block injected into the DYNAMIC part of every prompt.

Rules:
- Goes into the DYNAMIC part (after the static prefix) so prompt caching works.
- Returns compact text only — no dates, no names that vary — so the STATIC prefix
  remains byte-identical across requests.
- The context block is used SPARINGLY by the LLM (at most one item per few messages).
- Festival dates loaded from festivals.json (2025-2028), verified against two sources.

Sources for festival dates:
  Primary   : drikpanchang.com (lunisolar computation)
  Secondary : timeanddate.com / testbook.com (cross-verified)
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# ── Load festival data once at module import ─────────────────────────────────
_FESTIVAL_FILE = Path(__file__).parent / "festivals.json"

try:
    with open(_FESTIVAL_FILE, encoding="utf-8") as _f:
        _FESTIVAL_DATA: list = json.load(_f)["festivals"]
except Exception as _e:
    logger.error("Failed to load festivals.json: %s", _e)
    _FESTIVAL_DATA = []


# ── Day-part thresholds ────────────────────────────────────────────────────────
_DAY_PARTS = [
    (5,  "early morning"),
    (7,  "morning"),       # sunrise
    (10, "late morning"),
    (12, "noon"),
    (14, "afternoon"),
    (17, "evening"),
    (20, "night"),
    (23, "late night"),
    (25, "late night"),    # sentinel
]

_MEAL_WINDOWS = [
    ((7, 10),   "breakfast time"),
    ((12, 15),  "lunch time"),
    ((16, 18),  "chai/snack time"),
    ((19, 22),  "dinner time"),
    ((23, 25),  "late-night craving time"),
]

_WEEKDAY_VIBES = {
    0: "Monday — fresh week, post-weekend sluggish",
    1: "Tuesday — getting into work mode",
    2: "Wednesday — midweek hump day",
    3: "Thursday — almost there, Friday feeling building",
    4: "Friday — TGIF energy, weekend plans talk",
    5: "Saturday — weekend chill or weekend work 😭",
    6: "Sunday — lazy Sunday, weekend ending vibes",
}

_MONTH_VIBES = {
    1:  "January — new year, cold winter, blanket weather",
    2:  "February — Valentine month, slightly warmer",
    3:  "March — Holi vibes, end of winter",
    4:  "April — summer starting, exams season",
    5:  "May — peak summer, mango season",
    6:  "June — pre-monsoon heat",
    7:  "July — monsoon, chai-pakoda weather",
    8:  "August — monsoon peak, Independence Day",
    9:  "September — post-monsoon, mild",
    10: "October — festive season begins (Navratri, Diwali approaching)",
    11: "November — post-Diwali, mild winter starting",
    12: "December — Christmas-New Year, cold, year-end vibes",
}


def _get_day_part(hour: int) -> str:
    for threshold, label in _DAY_PARTS:
        if hour < threshold:
            return label
    return "late night"


def _get_meal_window(hour: int) -> Optional[str]:
    for (start, end), label in _MEAL_WINDOWS:
        if start <= hour < end:
            return label
    return None


def _compute_festival_context(today: date) -> str:
    """Return a short text about near-term festivals (within ±14 days)."""
    hits = []
    for fest in _FESTIVAL_DATA:
        name = fest["name"]
        hint = fest.get("topic_hint") or fest.get("topic_hindi", "")

        # Each festival entry can have a single "date" or "start"/"end"
        for entry in fest.get("dates", []):
            if "date" in entry:
                fest_start = date.fromisoformat(entry["date"])
                fest_end = fest_start
            elif "start" in entry and "end" in entry:
                fest_start = date.fromisoformat(entry["start"])
                fest_end = date.fromisoformat(entry["end"])
            else:
                continue

            delta_start = (fest_start - today).days
            delta_end   = (fest_end - today).days

            # Ongoing
            if fest_start <= today <= fest_end:
                hits.append(
                    f"• {name} is CURRENTLY HAPPENING (ends in {delta_end} day(s)). "
                    f"Topic hint: {hint}"
                )
            # Upcoming within 14 days
            elif 0 < delta_start <= 14:
                hits.append(
                    f"• {name} is coming up in {delta_start} day(s) ({fest_start.strftime('%d %b')}). "
                    f"Topic hint: {hint}"
                )
            # Just passed within 5 days
            elif -5 <= delta_end < 0:
                hits.append(
                    f"• {name} just ended {abs(delta_end)} day(s) ago. "
                    f"Topic hint (memories/recap): {hint}"
                )
    return "\n".join(hits) if hits else ""


def build_live_context(timezone: str = "Asia/Kolkata") -> str:
    """
    Build the compact [LIVE CONTEXT] block for injection into the DYNAMIC prompt section.
    This changes per request so it MUST NOT go into the static prefix.
    """
    try:
        tz = ZoneInfo(timezone)
    except Exception:
        tz = ZoneInfo("Asia/Kolkata")

    from datetime import datetime
    now = datetime.now(tz)
    today = now.date()
    hour = now.hour
    weekday = now.weekday()  # 0=Mon

    day_part    = _get_day_part(hour)
    meal_window = _get_meal_window(hour)
    weekday_vibe = _WEEKDAY_VIBES.get(weekday, "")
    month_vibe   = _MONTH_VIBES.get(today.month, "")
    festival_ctx = _compute_festival_context(today)

    lines = [
        f"[LIVE CONTEXT — inject sparingly, at most ONE item every few messages]",
        f"Date & time : {today.strftime('%A, %d %B %Y')}, {now.strftime('%I:%M %p')} {timezone}",
        f"Part of day : {day_part}",
    ]
    if meal_window:
        lines.append(f"Meal window : {meal_window}")
    if weekday_vibe:
        lines.append(f"Day vibe    : {weekday_vibe}")
    if month_vibe:
        lines.append(f"Season vibe : {month_vibe}")
    if festival_ctx:
        lines.append(f"Festival calendar :\n{festival_ctx}")

    lines.append(
        "Use context naturally like a real person — do NOT repeat festival/time/meal talk "
        "if it came up in the last ~10 messages. Do NOT quote the exact clock time unless asked."
    )

    return "\n".join(lines)


def get_proactive_opener_context(timezone: str = "Asia/Kolkata") -> dict:
    """
    Returns context dict for the proactive opener generator.
    Called when user has been idle for X hours or returns at a new part of day.
    """
    try:
        tz = ZoneInfo(timezone)
    except Exception:
        tz = ZoneInfo("Asia/Kolkata")

    from datetime import datetime
    now = datetime.now(tz)
    today = now.date()
    hour = now.hour

    return {
        "day_part": _get_day_part(hour),
        "meal_window": _get_meal_window(hour),
        "weekday": now.strftime("%A"),
        "date": today.isoformat(),
        "festival_ctx": _compute_festival_context(today),
        "month_vibe": _MONTH_VIBES.get(today.month, ""),
    }
