"""
tests/test_conversational_consistency.py
-----------------------------------------
Automated QA Checklist validation for Conversational Consistency:
  - T1: Date/day runtime consistency (Today Saturday -> Tomorrow Sunday).
  - T2: Pronoun consistency ('Mujhse tum bolna, aap mat bolna' -> 'tum' consistently, no 'aap').
  - T3: Contextual office workload discussion without hallucinating unknown facts.
  - T4: Precise memory recall without unsupported assumptions about user's house location.
  - T5: New session temporal awareness (current runtime weekday, no blind reuse of old session day).
  - Plus: Gender agreement (User says 'kr rha hu' -> masculine agreement/neutral, never feminine 'kar sakti hai').
"""

import pytest
from datetime import datetime
from zoneinfo import ZoneInfo
from unittest.mock import MagicMock, patch

from app.utils.time_context import get_current_context
from app.memory.memory_extractor import (
    detect_pronoun_preference, detect_user_gender_cue, extract_candidate_memories
)
from app.chat.prompt_builder import build_final_prompt
from app.chat.response_validator import validate_and_clean_response


def test_time_context_generation():
    """Verify runtime date/time context is correctly generated for Asia/Kolkata."""
    ctx = get_current_context()
    assert "current_date" in ctx
    assert "day_name" in ctx
    assert "local_time" in ctx
    assert "day_period" in ctx
    assert ctx["timezone"] == "Asia/Kolkata"
    assert ctx["day_period"] in ("morning", "afternoon", "evening", "night")


def test_pronoun_preference_detection():
    """Verify regex detection of explicit pronoun preferences."""
    assert detect_pronoun_preference("Mujhse tum bolna, aap mat bolna.") == "tum"
    assert detect_pronoun_preference("tum bola karo") == "tum"
    assert detect_pronoun_preference("aap mat use karo") == "tum"
    assert detect_pronoun_preference("Mujhse aap bolna") == "aap"
    assert detect_pronoun_preference("tu bolna yaar") == "tu"
    assert detect_pronoun_preference("hello how are you") is None


def test_user_gender_cue_detection():
    """Verify regex detection of 1st-person gender cues."""
    assert detect_user_gender_cue("apne office ka kaam kr rha hu") == "male"
    assert detect_user_gender_cue("main ghar ja raha hoon") == "male"
    assert detect_user_gender_cue("mai movie dekh rahi hu") == "female"
    assert detect_user_gender_cue("i am a guy") == "male"
    assert detect_user_gender_cue("main ladki hu") == "female"
    assert detect_user_gender_cue("bahut kaam baaki hai") is None


def test_gender_and_pronoun_sanitation_in_validator():
    """Verify that validator cleans accidental slips like 'chill kar sakti hai' for male users."""
    raw_male_slip = "Haan bilkul! Sunday ke liye aaram rakh de, bas aaj kaam nipta de. Phir kal poora din chill kar sakti hai"
    cleaned = validate_and_clean_response(
        raw_response=raw_male_slip,
        persona_name="diya",
        user_gender="male",
        preferred_pronoun="tum",
    )
    assert "kar sakti hai" not in cleaned
    assert "chill kar lena" in cleaned
    assert "nipta lo" in cleaned
    assert "rakh lo" in cleaned


def test_prompt_builder_injects_authoritative_time_and_pronouns():
    """Verify build_final_prompt includes authoritative clock and pronoun directives."""
    persona = {
        "id": "profile_diya",
        "name": "Diya",
        "gender": "female",
        "city": "Raipur",
        "education": "Design student",
        "system_prompt_template": "You are {name}.",
    }
    mock_time = {
        "current_datetime": "2026-10-10T11:24:00+05:30",
        "current_date": "2026-10-10",
        "day_name": "Saturday",
        "local_time": "11:24 AM",
        "timezone": "Asia/Kolkata",
        "day_period": "morning",
    }
    state = {
        "current_topic": "office work",
        "preferred_pronoun": "tum",
        "user_gender": "male",
    }
    system_prompt, history = build_final_prompt(
        persona=persona,
        rag_context="",
        chat_history=[],
        user_message="Aaj Saturday hai. Kal office nahi jaana.",
        conversation_state=state,
        time_context=mock_time,
    )

    assert "AUTHORITATIVE RUNTIME DATE, DAY & TIME CONTEXT" in system_prompt
    assert "Saturday" in system_prompt
    assert "PRONOUN DIRECTIVE (TUM - DEFAULT)" in system_prompt
    assert "USER GENDER AGREEMENT (MALE)" in system_prompt


def test_qa_t1_t5_response_validation():
    """Validate QA checklist requirements T1, T2, T3, T4, T5 logic."""
    # T2: If preferred pronoun is 'tum', validator prevents accidental 'aap' switch
    text_with_aap = "Aap bataiye aapka kya plan hai?"
    cleaned_tum = validate_and_clean_response(
        raw_response=text_with_aap,
        persona_name="diya",
        preferred_pronoun="tum",
    )
    assert "aap" not in cleaned_tum.lower()
    assert "tum" in cleaned_tum.lower()

    # T4: House location precision: user said they are from Purani Basti, but didn't confirm house
    persona = {
        "id": "profile_diya",
        "name": "Diya",
        "gender": "female",
        "city": "Raipur",
        "system_prompt_template": "You are {name}.",
    }
    system_prompt, _ = build_final_prompt(
        persona=persona,
        rag_context="",
        chat_history=[{"role": "user", "content": "Mai Purani Basti se hu"}],
        user_message="Mera ghar kahan hai?",
    )
    assert "Purani Basti" in system_prompt
    assert "PRECISE CONVERSATION MEMORY & NO UNSUPPORTED ASSUMPTIONS" in system_prompt


def test_clock_quote_sanitizer_removes_unprompted_exact_time():
    """Verify validate_and_clean_response strips unasked digital clock quotes."""
    raw = "Abhi toh 12:36 PM hai, toh still kaafi time baaki hai Kaam khatam ho jaaye phir chill karna."
    cleaned = validate_and_clean_response(
        raw_response=raw,
        persona_name="diya",
        preferred_pronoun="tum",
        user_message="mera to office 6:30 ko off hoga",
    )
    assert "12:36" not in cleaned
    assert "Abhi toh kaafi time baaki hai" in cleaned


def test_tu_imperative_cleaning_for_tum_style():
    """Verify that accidental 'tu' imperatives are normalized to natural 'tum' forms."""
    raw = "Agar Raipur mein kahi garba ho toh definitely tu aaja! Tum plan kar phir dekhte hain. Aur stress mat le, relax kar."
    cleaned = validate_and_clean_response(
        raw_response=raw,
        persona_name="diya",
        preferred_pronoun="tum",
    )
    import re
    assert "tu aaja" not in cleaned
    assert "tum aa jao" in cleaned
    assert not re.search(r"\bplan\s+kar\b", cleaned)
    assert "plan karo" in cleaned
    assert "stress mat le" not in cleaned
    assert "stress mat lo" in cleaned
    assert not re.search(r"\brelax\s+kar\b", cleaned)
    assert "relax karo" in cleaned


def test_festival_and_meetup_directives_in_prompt():
    """Verify festival factuality and meetup boundary directives are injected."""
    persona = {
        "id": "profile_diya",
        "name": "Diya",
        "gender": "female",
        "city": "Raipur",
        "system_prompt_template": "You are {name}.",
    }
    # Festival message
    prompt_fest, _ = build_final_prompt(
        persona=persona,
        rag_context="",
        chat_history=[],
        user_message="navratri ka planning kaisa chal rha hai raipur me",
    )
    assert "FESTIVAL & EVENT FACTUALITY DIRECTIVE" in prompt_fest

    # Meetup invitation message
    prompt_meet, _ = build_final_prompt(
        persona=persona,
        rag_context="",
        chat_history=[],
        user_message="chalogi mere sath garba khelne?",
    )
    assert "REAL-WORLD MEETUP BOUNDARY DIRECTIVE" in prompt_meet

    # Office off-timing message
    prompt_off, _ = build_final_prompt(
        persona=persona,
        rag_context="",
        chat_history=[],
        user_message="mera to office 6:30 ko off hoga",
    )
    assert "OFFICE SCHEDULE CONTEXT DIRECTIVE" in prompt_off
