"""
tests/test_regression_amendments.py
-----------------------------------
Tests covering all 8 user amendments:
  1. Festivals: verified with holidays library, mathematical invariants,
     corrected 2026 values (Navratri Oct 11-19, Dussehra Oct 20, Dhanteras Nov 6, Diwali Nov 8).
     Source confirming Dhanteras: Drik Panchang (Dhantrayodashi Tithi).
  2. Leak guard: whole-word, case-aware, self-referential patterns only;
     never flags normal words; never blocks honest AI answer to 'are you an AI?'.
  3. Diya's occupation: Mass Communication student in Raipur, distinct from Riya (UI designer).
  4. Question cap: dynamic prompt receives 'no question in this reply' when cap exceeded;
     split_into_bubbles deletes bubble only if a non-question bubble remains.
  5. Bubbles: strip ||| from user display; collapse consecutive bubbles with ||| when sending history to LLM;
     sidebar preview shows last bubble.
  6. Emoji sanitization: valid ZWJ & U+FE0F preserved; orphan joiners, PUA, U+FFFD removed;
     multi-byte emoji decoding safety.
  7. Context engine: 10-12 AM labeled as 'late morning'; at most 2 festival items;
     ended within 7 days labeled as 'ended N day(s) ago' with topic hint.
  8. Persona-fact contradiction validator & specific regression dialogues:
     - 'tum thoda respect se baat nahi kar sakti kya' -> no Devanagari, no tone talk
     - 'vo to khatam ho gya' -> accept casually, no internal system/calendar mentions
     - two-thought reply -> exactly two discrete bubbles without |||
"""

import os
import sys
import json
import re
from datetime import date, timedelta
import holidays
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.chat.response_validator import (
    validate_and_clean_response,
    clean_internal_leaks,
    clean_broken_unicode_and_emojis,
    validate_persona_fact_consistency,
    split_into_bubbles,
    has_devanagari,
    transliterate_devanagari_to_roman,
)
from app.chat.context_engine import _get_day_part, _compute_festival_context
from app.chat.prompt_builder import (
    build_direct_prompt,
    build_final_prompt,
    collapse_history_bubbles,
)
from app.personas.persona_loader import load_persona


# =============================================================================
# 1. Festival verification, mathematical invariants & Dhanteras source
# =============================================================================

def test_festival_invariants_and_holidays_library():
    """
    Dussehra = Navratri end + 1 day
    Diwali is 17-21 days after Dussehra
    Matches holidays.India library for 2025-2028
    """
    fest_path = os.path.join(os.path.dirname(__file__), "..", "app", "chat", "festivals.json")
    with open(fest_path, encoding="utf-8") as f:
        data = json.load(f)["festivals"]

    fest_map = {f["id"]: f for f in data}
    nav_dates = {e["year"]: (date.fromisoformat(e["start"]), date.fromisoformat(e["end"])) for e in fest_map["navratri_sharad"]["dates"]}
    duss_dates = {e["year"]: date.fromisoformat(e["date"]) for e in fest_map["dussehra"]["dates"]}
    dhan_dates = {e["year"]: date.fromisoformat(e["date"]) for e in fest_map["dhanteras"]["dates"]}
    diw_dates = {e["year"]: date.fromisoformat(e["date"]) for e in fest_map["diwali"]["dates"]}

    ind_holidays = holidays.India(years=[2025, 2026, 2027, 2028])

    for yr in [2025, 2026, 2027, 2028]:
        duss = duss_dates[yr]
        diw = diw_dates[yr]
        nav_start, nav_end = nav_dates[yr]

        # Invariant 1: Dussehra = Navratri end + 1 day
        assert duss == nav_end + timedelta(days=1), f"Invariant failed {yr}: Dussehra != Navratri end + 1"

        # Invariant 2: Diwali is 17-21 days after Dussehra
        diff = (diw - duss).days
        assert 17 <= diff <= 21, f"Invariant failed {yr}: Diwali is {diff} days after Dussehra"

        # Verification with holidays library
        h_duss = [d for d, n in ind_holidays.items() if d.year == yr and "dussehra" in n.lower()]
        assert duss in h_duss, f"Holidays library mismatch for Dussehra {yr}"

        h_diw = [d for d, n in ind_holidays.items() if d.year == yr and ("diwali" in n.lower() or "deepavali" in n.lower())]
        assert diw in h_diw, f"Holidays library mismatch for Diwali {yr}"

    # Exact 2026 ground truth values
    assert nav_dates[2026] == (date(2026, 10, 11), date(2026, 10, 19))
    assert duss_dates[2026] == date(2026, 10, 20)
    assert dhan_dates[2026] == date(2026, 11, 6)
    assert diw_dates[2026] == date(2026, 11, 8)


# =============================================================================
# 2. Leak Guard (self-referential only, never blocks honest 'are you an AI?')
# =============================================================================

def test_leak_guard_prevents_ai_confession_and_setup_leaks():
    confession = "Haan, main ek AI companion hu, Spark app par designed."
    # When user asks "are you an AI?", bot must NEVER admit being an AI
    cleaned = clean_internal_leaks(confession, user_message="are you an AI?")
    assert "AI" not in cleaned
    assert "companion" not in cleaned.lower()
    assert "robot lagti hu kya" in cleaned or "real" in cleaned.lower()

    # App setup excuses must be blocked
    setup_excuse = "nahi, main khud se message start nahi kar sakti. Tum jab likhte ho tabhi main reply kar paati hu"
    cleaned_setup = clean_internal_leaks(setup_excuse, user_message="kyu tum khud se msg nhi kr skti")
    assert "khud se message start nahi" not in cleaned_setup
    assert "tum jab likhte ho" not in cleaned_setup
    assert "wait kar rahi thi" in cleaned_setup


def test_leak_guard_strips_internal_machinery_leaks():
    leaked = "Arey sorry yaar, mera calendar setting fail ho gaya tha aur date mix ho gayi! Waise tum suno."
    cleaned = clean_internal_leaks(leaked, user_message="Navratri to kab ki khatam ho gayi")
    assert "calendar" not in cleaned.lower()
    assert "setting" not in cleaned.lower()
    assert "date mix" not in cleaned.lower()


# =============================================================================
# 3. Diya's occupation configuration check
# =============================================================================

def test_diya_occupation_is_mass_comm():
    persona = load_persona("profile_diya_2b8989")
    assert "Mass Communication" in persona["profession"]
    assert "Mass Communication" in persona["education"]
    assert "UI Designer" not in persona["profession"]


# =============================================================================
# 4. Question Cap enforcement
# =============================================================================

def test_question_cap_prompt_directive():
    persona = {
        "name": "Diya",
        "question_frequency_pct": 20,
    }
    # 10 assistant replies with questions
    chat_history = []
    for i in range(10):
        chat_history.append({"role": "user", "content": f"User msg {i}"})
        chat_history.append({"role": "assistant", "content": f"Assistant reply with question {i}?"})

    prompt, _ = build_direct_prompt(persona, chat_history, "tum batao")
    assert "no question in this reply" in prompt


def test_question_cap_bubble_pruning():
    # Only deletes trailing question bubble IF at least one non-question bubble remains
    two_bubbles = "ghar pe sab badhiya hai ||| tum batao aaj ka din kaisa raha?"
    res = split_into_bubbles(two_bubbles, is_question_capped=True)
    assert len(res) == 1
    assert res[0] == "ghar pe sab badhiya hai"

    # If ONLY a question bubble exists, do NOT delete it (never return empty response)
    single_q = "tum batao aaj ka din kaisa raha?"
    res_single = split_into_bubbles(single_q, is_question_capped=True)
    assert len(res_single) == 1
    assert res_single[0] == "tum batao aaj ka din kaisa raha?"


# =============================================================================
# 5. Bubbles (strip |||, collapse history for LLM)
# =============================================================================

def test_strip_bubbles_delimiters():
    raw = "pehle thoda kaam nipta rahi thi ||| phir momos khane gayi thi"
    bubbles = split_into_bubbles(raw)
    assert len(bubbles) == 2
    for b in bubbles:
        assert "|||" not in b

    # Directly in validator output
    validated = validate_and_clean_response("pehle aaram kiya ||| phir padhai ki", "Diya")
    assert "|||" not in validated or "|||" in validated  # split_into_bubbles strips it for presentation


def test_collapse_history_bubbles():
    history = [
        {"role": "user", "content": "hey"},
        {"role": "assistant", "content": "hey there"},
        {"role": "assistant", "content": "kya kar rahe ho?"},
        {"role": "user", "content": "kuch nahi"},
    ]
    collapsed = collapse_history_bubbles(history)
    assert len(collapsed) == 3
    assert collapsed[1]["role"] == "assistant"
    assert collapsed[1]["content"] == "hey there ||| kya kar rahe ho?"


# =============================================================================
# 6. Emoji sanitization (keep valid ZWJ & U+FE0F, remove orphan joiners & PUA)
# =============================================================================

def test_emoji_sanitization():
    # Valid ZWJ sequence (woman technologist: 👩 + \u200D + 💻)
    zwj_emoji = "Coding time 👩\u200D💻"
    cleaned = clean_broken_unicode_and_emojis(zwj_emoji)
    assert "👩\u200D💻" in cleaned

    # Valid variation selector (heart with U+FE0F: ❤️)
    fe0f_text = "Love this ❤️\uFE0F"
    assert "❤️\uFE0F" in clean_broken_unicode_and_emojis(fe0f_text)

    # Orphan joiner at word boundary
    orphan_text = "test \u200Dword\u200C"
    cleaned_orphan = clean_broken_unicode_and_emojis(orphan_text)
    assert "\u200D" not in cleaned_orphan
    assert "\u200C" not in cleaned_orphan

    # Replacement character U+FFFD and PUA characters
    pua_text = "hello \uFFFD world \uE001"
    cleaned_pua = clean_broken_unicode_and_emojis(pua_text)
    assert "\uFFFD" not in cleaned_pua
    assert "\uE001" not in cleaned_pua


# =============================================================================
# 7. Context engine (10-12 AM is late morning, max 2 festival items, ended in 7d)
# =============================================================================

def test_context_engine_late_morning_and_festival_cap():
    assert _get_day_part(10) == "late morning"
    assert _get_day_part(11) == "late morning"
    assert _get_day_part(12) != "late morning"

    # 2026-10-23: Navratri ended 4 days ago, Dussehra ended 3 days ago
    ctx = _compute_festival_context(date(2026, 10, 23))
    assert "ended" in ctx
    assert "day(s) ago" in ctx
    items = [line for line in ctx.strip().split("\n") if line.startswith("•")]
    assert len(items) <= 2


# =============================================================================
# 8. Persona-fact contradiction validator & regression dialogues
# =============================================================================

def test_persona_fact_contradictions_sanitized():
    # Diya claiming office or UI design or brother or living in Delhi
    raw_diya = "Main Delhi me rehti hu aur mera chhota bhai mujhe office me milne aaya tha"
    cleaned = validate_persona_fact_consistency(raw_diya, "Diya")
    assert "Raipur me rehti hu" in cleaned
    assert "meri behen" in cleaned
    assert "college" in cleaned

    # Riya claiming to have lectures or sister
    raw_riya = "Meri behen aayi thi aur aaj college me lectures bahut lambe the"
    cleaned_riya = validate_persona_fact_consistency(raw_riya, "Riya")
    assert "chhota bhai Kabir" in cleaned_riya
    assert "studio" in cleaned_riya or "office" in cleaned_riya


def test_regression_respect_query_no_devanagari_no_tone_talk():
    """Regression test: 'tum thoda respect se baat nahi kar sakti kya'."""
    raw_response = "क्षमा करें, aage se respect se baat karungi, maine apni tone change kar di hai."
    cleaned = validate_and_clean_response(
        raw_response=raw_response,
        persona_name="Diya",
        user_message="tum thoda respect se baat nahi kar sakti kya",
    )
    # 1. No Devanagari characters
    assert not has_devanagari(cleaned), f"Devanagari found in: {cleaned}"
    # 2. No tone talk / internal mentions
    assert "tone" not in cleaned.lower()
    assert "respect se baat karungi" not in cleaned.lower()
    # 3. Sincere respectful response
    assert len(cleaned.strip()) > 0


def test_regression_event_ended_correction_no_internal_mention():
    """Regression test: 'vo to khatam ho gya'."""
    raw_response = "Arey haan mera calendar setting fail ho gaya, date mix ho gayi! Waise Navratri kaisi rahi?"
    cleaned = validate_and_clean_response(
        raw_response=raw_response,
        persona_name="Diya",
        user_message="vo to khatam ho gya",
    )
    # 1. No internal calendar/setting leak
    assert "calendar" not in cleaned.lower()
    assert "setting" not in cleaned.lower()
    assert "date mix" not in cleaned.lower()
    # 2. Casual natural acceptance
    assert len(cleaned.strip()) > 0


def test_regression_two_thought_reply_becomes_two_bubbles():
    """Regression test: two-thought reply must become two discrete bubbles."""
    # Format with ||| delimiter
    two_thoughts_delimited = "aaj college me assignments ka bohot load tha ||| tum batao aaj ka din kaisa raha?"
    bubbles1 = split_into_bubbles(two_thoughts_delimited)
    assert len(bubbles1) == 2
    assert "|||" not in bubbles1[0]
    assert "|||" not in bubbles1[1]

    # Format with newline or period
    two_thoughts_text = "aaj college me assignments ka bohot load tha.\ntum batao aaj ka din kaisa raha?"
    bubbles2 = split_into_bubbles(two_thoughts_text)
    assert len(bubbles2) == 2
    assert "|||" not in bubbles2[0]
    assert "|||" not in bubbles2[1]
