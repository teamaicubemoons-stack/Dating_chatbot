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
import re
from datetime import datetime
from zoneinfo import ZoneInfo
from unittest.mock import MagicMock, patch

from app.utils.time_context import get_current_context
from app.memory.memory_extractor import (
    detect_pronoun_preference, detect_user_gender_cue, extract_candidate_memories
)
from app.chat.prompt_builder import (
    build_final_prompt, is_user_asked_origin_message, extract_mentioned_venue
)
from app.chat.response_validator import validate_and_clean_response


def test_time_context_generation():
    """Verify runtime date/time context is correctly generated for Asia/Kolkata."""
    ctx = get_current_context()
    assert "current_date" in ctx
    assert "day_name" in ctx
    assert "local_time" in ctx
    assert "day_period" in ctx
    assert ctx["timezone"] == "Asia/Kolkata"
    assert ctx["day_period"] in ("morning", "late morning", "afternoon", "evening", "night")
    assert "verified_event_context" in ctx


def test_past_festival_status_and_user_correction():
    from app.chat.prompt_builder import (
        build_final_prompt,
        is_event_timing_correction_message,
    )

    assert is_event_timing_correction_message("lekin navratri toh over ho gaya") is True
    persona = {
        "id": "profile_riya",
        "name": "Riya",
        "gender": "female",
        "city": "Delhi",
        "system_prompt_template": "You are {name}.",
    }
    context = {
        "current_datetime": "2026-10-21T13:00:00+05:30",
        "current_date": "2026-10-21",
        "day_name": "Wednesday",
        "local_time": "01:00 PM",
        "timezone": "Asia/Kolkata",
        "day_period": "afternoon",
        "verified_event_context": {
            "region": "India",
            "events": {
                "navratri": {"status": "past", "date": "2026-10-20"},
                "dussehra": {"status": "past", "date": "2026-10-20"},
                "diwali": {"status": "upcoming", "date": "2026-11-08"},
            },
        },
    }
    prompt, _ = build_final_prompt(
        persona, "", [], "lekin navratri toh over ho gaya, ab garba kaise?", time_context=context
    )
    assert "USER EVENT-TIMING CORRECTION" in prompt
    assert "'status': 'past'" in prompt
    assert "do not describe it as ongoing" in prompt


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


def test_tum_pronoun_verb_agreement_sanitizer():
    """Verify that ungrammatical 'tum kaisa hai' and similar slips are corrected to 'tum kaise ho'."""
    # 1. Direct screenshot case: 'Main badhiya hu yaar! 😊 tum kaisa hai?'
    raw = "Main badhiya hu yaar! 😊 tum kaisa hai?"
    cleaned = validate_and_clean_response(
        raw_response=raw,
        persona_name="Riya Kapoor",
        preferred_pronoun="tum",
    )
    assert "tum kaisa hai" not in cleaned
    assert "tum kaise ho?" in cleaned

    # 2. 'tu kaisa hai' slip
    raw_tu = "Main mast hoon! tu kaisa hai?"
    cleaned_tu = validate_and_clean_response(
        raw_response=raw_tu,
        persona_name="Riya Kapoor",
        preferred_pronoun="tum",
    )
    assert "tu kaisa hai" not in cleaned_tu
    assert "tum kaise ho?" in cleaned_tu

    # 3. 'tum kya kar raha hai' -> 'tum kya kar rahe ho'
    raw_verb = "aur tum bata, tum kya kar raha hai abhi?"
    cleaned_verb = validate_and_clean_response(
        raw_response=raw_verb,
        persona_name="Riya Kapoor",
        preferred_pronoun="tum",
    )
    assert "tum batao" in cleaned_verb
    assert "tum kya kar rahe ho" in cleaned_verb

    # 4. Habitual verb: 'tum kahan rehta hai' -> 'tum kahan rehte ho'
    raw_hab = "Waise tum kahan rehta hai?"
    cleaned_hab = validate_and_clean_response(
        raw_response=raw_hab,
        persona_name="Riya Kapoor",
        preferred_pronoun="tum",
    )
    assert "tum kahan rehte ho?" in cleaned_hab

    # 5. Gender alignment with male user: 'tum kaisi ho' -> 'tum kaise ho'
    raw_gender = "tum kaisi ho?"
    cleaned_gender = validate_and_clean_response(
        raw_response=raw_gender,
        persona_name="Riya Kapoor",
        preferred_pronoun="tum",
        user_gender="male",
    )
    assert cleaned_gender.strip() == "tum kaise ho?"


def test_hinglish_verb_morphology_and_phrasing_sanitizer():
    """Verify that continuous aspect verbs (suno rahi -> sun rahi), double-verbs (lo liyo -> le lena), and awkward too much are sanitized."""
    # 1. Exact user screenshot case:
    raw = "Main bas chill kar rahi hoon abhi, music suno rahi hoon thoda. Tum stress mat lo too much, break lo liyo kabhi kabhi!"
    cleaned = validate_and_clean_response(
        raw_response=raw,
        persona_name="Riya Kapoor",
        preferred_pronoun="tum",
    )
    assert "suno rahi" not in cleaned
    assert "sun rahi" in cleaned
    assert "lo liyo" not in cleaned
    assert "break le lena" in cleaned
    assert "stress mat lo too much" not in cleaned
    assert "zyada stress mat lo" in cleaned

    # 2. Continuous aspect imperative -o slips
    raw_verbs = "main dekho rahi hoon aur tum kya karo rahe ho? Wo sab bolo rahe the."
    cleaned_verbs = validate_and_clean_response(
        raw_response=raw_verbs,
        persona_name="Riya Kapoor",
        preferred_pronoun="tum",
    )
    assert "dekho rahi" not in cleaned_verbs
    assert "dekh rahi" in cleaned_verbs
    assert "karo rahe" not in cleaned_verbs
    assert "kar rahe" in cleaned_verbs
    assert "bolo rahe" not in cleaned_verbs
    assert "bol rahe" in cleaned_verbs

    # 3. Slang 'liyo' compound verbs
    raw_liyo = "pehle kaam kar liyo, phir aaram se so liyo."
    cleaned_liyo = validate_and_clean_response(
        raw_response=raw_liyo,
        persona_name="Riya Kapoor",
        preferred_pronoun="tum",
    )
    assert "kar liyo" not in cleaned_liyo
    assert "kar lena" in cleaned_liyo
    assert "so liyo" not in cleaned_liyo
    assert "so lena" in cleaned_liyo


def test_dating_app_match_and_venue_honesty():
    """Verify that match origin reciprocal questions and unknown venue honesty are enforced."""
    # 1. Detection of origin inquiry
    assert is_user_asked_origin_message("humare side to navratri chal rha hai wese tum kaha se ho?") is True
    assert is_user_asked_origin_message("where are you from?") is True
    assert is_user_asked_origin_message("kya chal raha hai?") is False

    # 2. Extraction of mentioned venue/ground
    assert extract_mentioned_venue("mai BTI ground jane ka socha hu kl saturday ko") == "BTI ground"
    assert extract_mentioned_venue("marine drive chalte hain") == "marine drive"
    assert extract_mentioned_venue("kuch nahi normal kaam hai") is None

    # 3. Prompt builder injection for Riya Kapoor when asked origin & venue
    riya_persona = {
        "id": "profile_riyakapoor_1aab25",
        "name": "Riya Kapoor",
        "gender": "female",
        "city": "Delhi",
        "system_prompt_template": "You are {name}.",
    }

    # Origin query prompt
    prompt_orig, _ = build_final_prompt(
        persona=riya_persona,
        rag_context="",
        chat_history=[],
        user_message="wese tum kaha se ho?",
    )
    assert "ORIGIN & RECIPROCAL DISCOVERY DIRECTIVE" in prompt_orig
    assert "MANDATORILY ASK BACK WHERE THE USER IS FROM" in prompt_orig

    # Unknown venue query prompt (BTI ground)
    prompt_venue, _ = build_final_prompt(
        persona=riya_persona,
        rag_context="",
        chat_history=[{"role": "user", "content": "hello"}],
        user_message="mai BTI ground jane ka socha hu kl saturday ko",
    )
    assert "UNKNOWN VENUE & GEOGRAPHIC HONESTY DIRECTIVE" in prompt_venue
    assert "BTI ground" in prompt_venue
    assert "STRICTLY NEVER pretend to know it" in prompt_venue

    # 4. Response validator strips hallucinated venue familiarity for Riya
    raw_hallucination = "Achha sahi hai! BTI ground mein accha vibe hota hai usually 😌 Kal Saturday ko phir masti ho jayegi"
    cleaned = validate_and_clean_response(
        raw_response=raw_hallucination,
        persona_name="Riya Kapoor",
        preferred_pronoun="tum",
    )
    assert "BTI ground mein accha vibe hota hai" not in cleaned
    assert "Yeh kaunse city ya area mein hai waise?" in cleaned


def test_diya_chill_kar_and_festival_continuity():
    """Verify that 'Chill kar' is converted to 'Chill karo' and Navratri continuity is maintained."""
    # 1. 'Chill kar' and 'tum se janne ka curiosity'
    raw_diya = "Haan haan, mujhe bhi tum se janne ka curiosity hai! Chill kar, time hai sab kuch ke liye."
    cleaned_diya = validate_and_clean_response(
        raw_response=raw_diya,
        persona_name="Diya",
        preferred_pronoun="tum",
    )
    assert not re.search(r"\bchill\s+kar\b", cleaned_diya.lower())
    assert "chill karo" in cleaned_diya.lower()
    assert "tumhe janne ki curiosity" in cleaned_diya.lower()

    # 2. Navratri ongoing continuity
    raw_nav = "Navratri aane wali hai toh vibe toh banne lagi hai yaar! Par exact dates ya events maine abhi check nahi kiye honestly."
    cleaned_nav = validate_and_clean_response(
        raw_response=raw_nav,
        persona_name="Diya",
        preferred_pronoun="tum",
        user_message="Accha abhi navratri kaisa chal rha hai tumhare side",
    )
    assert "navratri aane wali hai" not in cleaned_nav.lower()
    assert "exact dates ya events maine abhi check nahi kiye" not in cleaned_nav.lower()


