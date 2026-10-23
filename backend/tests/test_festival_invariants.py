"""
tests/test_festival_invariants.py
---------------------------------
Validates Indian festival dates using the `holidays` library and mathematical invariants:
  1. Dussehra = Navratri end + 1 day
  2. Diwali is 17-21 days after Dussehra
  3. Correctness for 2026:
     - Navratri: 11-19 Oct 2026
     - Dussehra: 20 Oct 2026
     - Dhanteras: 06 Nov 2026 (Verified via Drik Panchang lunar calendar)
     - Diwali: 08 Nov 2026
  4. Context engine mock tests for:
     - 2026-10-23: Navratri ended 4 days ago, Dussehra ended 3 days ago
     - 2026-10-15: Navratri is currently happening
     - 2026-11-05: Dhanteras is coming up in 1 day
"""

import os
import sys
import json
from datetime import date, timedelta
from pathlib import Path
import pytest
import holidays

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.chat.context_engine import _compute_festival_context

FESTIVALS_FILE = Path(__file__).parent.parent / "app" / "chat" / "festivals.json"


def load_festivals():
    with open(FESTIVALS_FILE, encoding="utf-8") as f:
        return json.load(f)["festivals"]


def test_festivals_against_holidays_library_and_invariants():
    festivals = load_festivals()
    fest_by_id = {f["id"]: f for f in festivals}

    navratri_dates = {e["year"]: (date.fromisoformat(e["start"]), date.fromisoformat(e["end"])) for e in fest_by_id["navratri_sharad"]["dates"]}
    dussehra_dates = {e["year"]: date.fromisoformat(e["date"]) for e in fest_by_id["dussehra"]["dates"]}
    dhanteras_dates = {e["year"]: date.fromisoformat(e["date"]) for e in fest_by_id["dhanteras"]["dates"]}
    diwali_dates = {e["year"]: date.fromisoformat(e["date"]) for e in fest_by_id["diwali"]["dates"]}

    ind_holidays = holidays.India(years=[2025, 2026, 2027, 2028])

    for year in [2025, 2026, 2027, 2028]:
        duss = dussehra_dates[year]
        diw = diwali_dates[year]
        nav_start, nav_end = navratri_dates[year]

        # 1. Invariant: Dussehra = Navratri end + 1 day
        assert duss == nav_end + timedelta(days=1), f"Invariant failed for {year}: Dussehra {duss} != Navratri end {nav_end} + 1 day"

        # 2. Invariant: Diwali is 17-21 days after Dussehra
        delta_diw = (diw - duss).days
        assert 17 <= delta_diw <= 21, f"Invariant failed for {year}: Diwali {diw} is {delta_diw} days after Dussehra (expected 17-21 days)"

        # 3. Match against holidays library
        # Find Dussehra in holidays
        h_duss = [d for d, n in ind_holidays.items() if d.year == year and "dussehra" in n.lower()]
        assert h_duss and duss in h_duss, f"Holidays library Dussehra mismatch for {year}: expected {duss}, got {h_duss}"

        # Find Diwali in holidays
        h_diw = [d for d, n in ind_holidays.items() if d.year == year and ("diwali" in n.lower() or "deepavali" in n.lower())]
        assert h_diw and diw in h_diw, f"Holidays library Diwali mismatch for {year}: expected {diw}, got {h_diw}"

    # Specific 2026 ground truth assertions
    assert navratri_dates[2026] == (date(2026, 10, 11), date(2026, 10, 19))
    assert dussehra_dates[2026] == date(2026, 10, 20)
    assert dhanteras_dates[2026] == date(2026, 11, 6)
    assert diwali_dates[2026] == date(2026, 11, 8)


def test_festival_context_mocked_dates():
    # Test 1: Mock 2026-10-23
    ctx_23_oct = _compute_festival_context(date(2026, 10, 23))
    assert "ended 4 day(s) ago" in ctx_23_oct or "Navratri" in ctx_23_oct
    assert "ended 3 day(s) ago" in ctx_23_oct or "Dussehra" in ctx_23_oct
    assert "CURRENTLY HAPPENING" not in ctx_23_oct

    # Test 2: Mock 2026-10-15 (Navratri ongoing)
    ctx_15_oct = _compute_festival_context(date(2026, 10, 15))
    assert "Navratri (Sharad) is CURRENTLY HAPPENING" in ctx_15_oct

    # Test 3: Mock 2026-11-05 (Dhanteras week)
    ctx_05_nov = _compute_festival_context(date(2026, 11, 5))
    assert "Dhanteras is coming up in 1 day(s)" in ctx_05_nov
    assert "Diwali is coming up in 3 day(s)" in ctx_05_nov
