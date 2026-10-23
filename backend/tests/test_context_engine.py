"""
tests/test_context_engine.py
----------------------------
Unit tests for the live context engine:
  - Day-part classification (morning, noon, evening, night)
  - Meal window detection (breakfast, lunch, chai, dinner)
  - Weekday and month vibes
  - Festival countdown computation from festivals.json
"""

from datetime import date
from app.chat.context_engine import (
    _get_day_part,
    _get_meal_window,
    _compute_festival_context,
    build_live_context,
    _FESTIVAL_DATA,
)


def test_day_parts():
    assert _get_day_part(4) == "early morning"
    assert _get_day_part(6) == "morning"
    assert _get_day_part(9) == "late morning"
    assert _get_day_part(11) == "noon"
    assert _get_day_part(13) == "afternoon"
    assert _get_day_part(16) == "evening"
    assert _get_day_part(19) == "night"
    assert _get_day_part(22) == "late night"


def test_meal_windows():
    assert _get_meal_window(8) == "breakfast time"
    assert _get_meal_window(13) == "lunch time"
    assert _get_meal_window(17) == "chai/snack time"
    assert _get_meal_window(20) == "dinner time"
    assert _get_meal_window(4) is None  # not a regular meal window


def test_festival_data_loaded():
    assert len(_FESTIVAL_DATA) >= 10, "festivals.json should contain at least 10 major festivals"


def test_festival_countdown_during_festival():
    # Test with Diwali 2026 date (Nov 8, 2026)
    diwali_date = date(2026, 11, 8)
    ctx = _compute_festival_context(diwali_date)
    assert "Diwali" in ctx
    assert "HAPPENING" in ctx or "day(s)" in ctx


def test_festival_countdown_before_festival():
    # 5 days before Diwali 2026
    test_date = date(2026, 11, 3)
    ctx = _compute_festival_context(test_date)
    assert "Diwali" in ctx
    assert "coming up in 5 day(s)" in ctx


def test_build_live_context_smoke():
    block = build_live_context("Asia/Kolkata")
    assert "[LIVE CONTEXT" in block
    assert "Part of day :" in block
    assert "Date & time :" in block


if __name__ == "__main__":
    test_day_parts()
    test_meal_windows()
    test_festival_data_loaded()
    test_festival_countdown_during_festival()
    test_festival_countdown_before_festival()
    test_build_live_context_smoke()
    print("[PASS] All context_engine tests passed!")
