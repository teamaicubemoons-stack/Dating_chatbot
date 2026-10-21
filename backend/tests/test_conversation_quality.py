"""
tests/test_conversation_quality.py
-----------------------------------
Automated Conversation Quality Audit & Regression Test Suite.
Follows the SPARK Conversation Quality Specification & User Review Feedback:
  TEST 1: Greeting relevance (no unprompted activity invention in prompt)
  TEST 2: Correction acknowledgement (broad semantic coverage & prompt guidance)
  TEST 3: Hinglish style (language directive prevents canned English openers)
  TEST 4: Direct AI identity (honest, casual disclosure without marketing boilerplate)
  TEST 5: Natural follow-up (flexible response length, no mandatory interrogation)
  TEST 6: Persona differentiation (Diya vs Luna distinctiveness)
  TEST 7: Language switching (English -> Hinglish -> English context adaptation)
  TEST 8: Validator regression safety (valid sentences are never corrupted)
  TEST 9: Repetition suppression across turns
  TEST 10: Multi-turn correction persistence in conversation state
"""

import pytest
from app.chat.prompt_builder import (
    build_final_prompt, is_ai_identity_inquiry, is_user_correction_message
)
from app.chat.response_validator import validate_and_clean_response
from app.personas.persona_loader import load_persona
from app.utils.language_detector import detect_language
from app.memory.memory_extractor import extract_candidate_memories, process_and_persist_memories
from app.api.dependencies import SessionLocal
from app.models.db_models import ConversationState, Message, UserMemory


def test_1_greeting_relevance_directive():
    """TEST 1: Input 'Mai badhiya tum btao' receives prompt guidance prohibiting unsolicited activity claims."""
    diya = load_persona("profile_diya_2b8989")
    system_prompt, _ = build_final_prompt(
        persona=diya,
        rag_context="",
        chat_history=[],
        user_message="Mai badhiya tum btao"
    )
    # The prompt must explicitly forbid unprompted offline activities like sketching/gym during greetings
    assert "NO UNPROMPTED OFFLINE ACTIVITIES" in system_prompt
    assert "Do NOT invent unsolicited real-world physical activities" in system_prompt


def test_2_correction_acknowledgement_broad_coverage():
    """TEST 2: Broad semantic coverage of user corrections (unasked details, directness, formality, advice)."""
    test_corrections = [
        "Maine to tumse pucha hi nhi ki tum kya kr rhi ho ye tum kyo bta rhe ho mujhe",
        "arey maine ye kab pucha?",
        "seedha jawab do na",
        "tum har baar extra kyu bolti ho?",
        "itna formal mat bano",
        "unasked advice mat do",
        "faltu mat bolo",
        "i didn't ask",
    ]
    for msg in test_corrections:
        assert is_user_correction_message(msg) is True, f"Failed to detect correction: '{msg}'"

    # Non-corrections should NOT be classified as corrections
    non_corrections = [
        "Main badhiya hoon tum batao",
        "Kal kya karogi?",
        "Mujhe pizza pasand hai",
        "Aaj bohot thakan ho gayi",
    ]
    for msg in non_corrections:
        assert is_user_correction_message(msg) is False, f"False positive correction: '{msg}'"

    # Verify directive is injected when correction occurs
    diya = load_persona("profile_diya_2b8989")
    system_prompt, _ = build_final_prompt(
        persona=diya,
        rag_context="",
        chat_history=[{"role": "user", "content": "Mai badhiya tum btao"}, {"role": "assistant", "content": "Main bhi badhiya hu"}],
        user_message="seedha jawab do na"
    )
    assert "CORRECTION ACKNOWLEDGEMENT" in system_prompt
    assert "chill kar" in system_prompt


def test_3_hinglish_style_directive():
    """TEST 3: Input 'Aaj bs thoda kaam niptaana hai fir rest' ensures Hinglish directives."""
    msg = "Aaj bs thoda kaam niptaana hai fir rest"
    lang = detect_language(msg)
    assert lang == "hinglish"

    diya = load_persona("profile_diya_2b8989")
    system_prompt, _ = build_final_prompt(
        persona=diya,
        rag_context="",
        chat_history=[],
        user_message=msg
    )
    assert "HINGLISH DETECTED" in system_prompt
    assert "NEVER prepend generic English openers" in system_prompt


def test_4_direct_ai_identity():
    """TEST 4: Identity questions require honest AI disclosure."""
    msg1 = "tumhare bato se esa lg rha hai mujhe ki tum real ni ho esa"
    msg2 = "Tum real ho ya AI?"
    assert is_ai_identity_inquiry(msg1) is True
    assert is_ai_identity_inquiry(msg2) is True

    diya = load_persona("profile_diya_2b8989")
    system_prompt, _ = build_final_prompt(
        persona=diya,
        rag_context="",
        chat_history=[],
        user_message=msg1
    )
    assert "AI IDENTITY TRANSPARENCY" in system_prompt
    assert "respond honestly" in system_prompt.lower() or "answer honestly" in system_prompt.lower()


def test_5_natural_followup():
    """TEST 5: Natural follow-up allows flexible response length without mandatory interrogation."""
    diya = load_persona("profile_diya_2b8989")
    system_prompt, _ = build_final_prompt(
        persona=diya,
        rag_context="",
        chat_history=[],
        user_message="Bas aaj thoda tired hu."
    )
    assert "Ask at most ONE question" in system_prompt or "mostly 1 to 2 short sentences" in system_prompt


def test_6_persona_differentiation():
    """TEST 6: Compare Diya vs Luna prompts — verify distinct personality backstories and styles."""
    diya = load_persona("profile_diya_2b8989")
    luna = load_persona("profile_1")

    prompt_diya, _ = build_final_prompt(diya, "", [], "Hey")
    prompt_luna, _ = build_final_prompt(luna, "", [], "Hey")

    assert "Raipur" in prompt_diya and "design" in prompt_diya
    assert "Brooklyn" in prompt_luna and "photographer" in prompt_luna
    assert prompt_diya != prompt_luna


def test_7_language_switching():
    """TEST 7: English -> Hinglish -> English adapts dynamically based on message and history context."""
    diya = load_persona("profile_diya_2b8989")

    # Turn 1: English
    assert detect_language("Hello, how are you?") == "english"

    # Turn 2: Hinglish
    assert detect_language("Kaise ho aap?") == "hinglish"

    # Turn 3: Short English greeting following Hinglish context carries Hinglish directive
    hist = [
        {"role": "user", "content": "Kaise ho aap?"},
        {"role": "assistant", "content": "Main badhiya hoon, aap batao?"}
    ]
    prompt_3, _ = build_final_prompt(diya, "", hist, "All good")
    assert "HINGLISH DETECTED" in prompt_3


def test_8_validator_regression_safety():
    """TEST 8: Validator intercepts AI confession AND does NOT distort legitimate sentences."""
    # Honest AI disclosure must not be rewritten into a false human claim.
    disclosure_hinglish = "haha, pakad liya 😅 haan, main AI hu. par baat toh naturally kar sakte hain na :)"
    cleaned_h = validate_and_clean_response(disclosure_hinglish, "Diya")
    assert "main ai hu" in cleaned_h.lower()

    disclosure_english = "Yeah, I'm an AI companion created to chat!"
    cleaned_e = validate_and_clean_response(disclosure_english, "Luna")
    assert "i'm an ai" in cleaned_e.lower()

    # Valid sentences with words like 'chill' or 'plan' must NOT be mangled or corrupted
    valid_sentence_1 = "aaj hum dost chill kar rahe the marine drive par"
    cleaned_1 = validate_and_clean_response(valid_sentence_1, "Diya")
    assert "chill kar rahe the" in cleaned_1

    valid_sentence_2 = "Sounds like a solid plan for our upcoming photography project!"
    cleaned_2 = validate_and_clean_response(valid_sentence_2, "Luna")
    assert "Sounds like a solid plan" in cleaned_2


def test_9_repetition_suppression():
    """TEST 9: Recent consecutive assistant greetings trigger the repetition guard."""
    luna = load_persona("profile_1")
    history = [
        {"role": "user", "content": "Hey Luna"},
        {"role": "assistant", "content": "Hey 🙂 Glad you're here."},
        {"role": "user", "content": "What's up?"},
        {"role": "assistant", "content": "Hey! Just relaxing."}
    ]
    prompt, _ = build_final_prompt(luna, "", history, "What music do you like?")
    assert "REPETITION GUARD" in prompt
    assert "Do NOT start this message with 'Hey' or 'Hi'" in prompt


@pytest.fixture
def test_db():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.models.db_models import Base
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_10_multi_turn_correction_persistence(test_db):
    """TEST 10: Correction is recorded in conversation state and persists into subsequent turn prompt contexts."""
    user_id = "test_persistence_user_42"
    profile_id = "profile_diya_2b8989"

    # Step 1: User makes a correction
    critique_msg = "seedha jawab do na, extra mat bolo"
    process_and_persist_memories(test_db, user_id, profile_id, critique_msg)

    # Step 2: Fetch state and verify active_style_feedback is recorded
    state_record = test_db.query(ConversationState).filter_by(user_id=user_id, profile_id=profile_id).first()
    assert state_record is not None
    assert "active_style_feedback" in (state_record.recent_context_json or "")

    # Next turn message receives active style preference in system prompt
    next_turn_prompt, _ = build_final_prompt(
        persona=load_persona(profile_id),
        rag_context="",
        chat_history=[{"role": "user", "content": critique_msg}, {"role": "assistant", "content": "Samajh gayi, seedha point pe aate hain"}],
        user_message="Aaj mausam kaisa hai?",
        conversation_state={"active_style_feedback": "User requested direct and concise responses without unsolicited details or commentary."}
    )

    assert "Active User Style Preference" in next_turn_prompt
    assert "direct and concise" in next_turn_prompt.lower()


def test_11_contradiction_acknowledgement_directive():
    """TEST 11: User pointing out a factual mismatch (design vs medical) triggers contradiction directive."""
    from app.chat.prompt_builder import is_user_contradiction_message

    test_contradictions = [
        "tum medical collage me kaise abhi to tumne kaha tha na ki tum design pdhti hu krke",
        "pehle toh tumne design student bola tha ab medical kyu bol rahi ho?",
        "dono baatein match nahi kar rahi",
        "abhi to tum bol rahi thi ki Raipur mein ho abhi kuch aur bol rahi",
        "you just said you study design how are you in medical college",
    ]
    for msg in test_contradictions:
        assert is_user_contradiction_message(msg) is True, f"Failed to detect contradiction: '{msg}'"

    # Non-contradictions
    assert is_user_contradiction_message("Mujhe design achhi lagti hai") is False

    diya = load_persona("profile_diya_2b8989")
    system_prompt, _ = build_final_prompt(
        persona=diya,
        rag_context="",
        chat_history=[
            {"role": "user", "content": "RIMS full form btao na"},
            {"role": "assistant", "content": "Rajiv Institute of Medical Sciences."}
        ],
        user_message="tum medical collage me kaise abhi to tumne kaha tha na ki tum design pdhti hu krke"
    )
    assert "CRITICAL CONTRADICTION & MISMATCH ACKNOWLEDGEMENT" in system_prompt
    assert "Dono baatein match nahi kar rahi" in system_prompt
    assert "NEVER invent a convoluted excuse" in system_prompt


def test_12_college_inquiry_boundary_directive():
    """TEST 12: Inquiring about specific college name triggers factual boundary preventing hallucinated institutions."""
    from app.chat.prompt_builder import is_college_inquiry_message

    inquiries = [
        "Okay kon se collage me ho?",
        "kaun sa college hai tumhara?",
        "which college do you attend?",
        "college ka naam kya hai?",
        "RIMS full form btao na",
    ]
    for q in inquiries:
        assert is_college_inquiry_message(q) is True, f"Failed to detect college inquiry: '{q}'"

    diya = load_persona("profile_diya_2b8989")
    system_prompt, _ = build_final_prompt(
        persona=diya,
        rag_context="",
        chat_history=[],
        user_message="Okay kon se collage me ho?"
    )
    assert "COLLEGE & EDUCATION CONVERSATIONAL GUIDANCE" in system_prompt
    assert "Do NOT output a canned or hardcoded sentence" in system_prompt
    assert "profile" in system_prompt.lower()


def test_13_validator_factual_safety_guard():
    """TEST 13: Response validator catches unprompted medical college assertions for Diya."""
    # When Diya hallucinates medical college without acknowledging mistake
    hallucinated = "Main Rajiv Institute of Medical Sciences mein padhti hoon."
    cleaned = validate_and_clean_response(hallucinated, persona_name="Diya")
    assert "design college" in cleaned.lower()
    assert "Rajiv Institute" not in cleaned

    # When Diya acknowledges the error legitimately, it should not be rewritten
    admitted = "Haha sahi pakda 😅 main medical nahi, design college mein hoon. Confuse ho gayi thi!"
    cleaned_valid = validate_and_clean_response(admitted, persona_name="Diya")
    assert "sahi pakda" in cleaned_valid


def test_profile_privacy_does_not_leak_configuration_language():
    from app.chat.prompt_builder import build_direct_prompt

    prompt, _ = build_direct_prompt(
        {"name": "Diya", "city": "Raipur", "short_bio": "Design student"},
        [],
        "Have you ever been to Delhi?",
    )
    assert "PROFILE PRIVACY" in prompt
    leaked = "Profile me ye mention nahi hai, par Delhi ke cafes explore karne ka mauka mile to zaroor jaungi!"
    cleaned = validate_and_clean_response(leaked, "Diya")
    assert "profile" not in cleaned.lower()
    assert "Delhi ke cafes" in cleaned


def test_14_bounded_timeout_settings():
    """TEST 14: Settings ensure bounded request timeouts and retries for latency guarantees."""
    from app.config.settings import settings
    assert settings.llm_request_timeout_seconds <= 15.0
    assert settings.llm_max_retries <= 1


def test_15_engaging_conversational_rhythm_and_short_reply_handling():
    """TEST 15: Prompt contains the React -> Connect -> Add Personality rhythm and short-reply guidance."""
    from app.chat.prompt_builder import build_direct_prompt, build_final_prompt
    from app.personas.persona_loader import load_persona

    riya = load_persona("profile_riyakapoor_1aab25")
    direct_prompt, _ = build_direct_prompt(
        persona=riya,
        chat_history=[
            {"role": "assistant", "content": "lunch abhi kaisa tha? koi special dish ya bas simple khana"},
        ],
        user_message="bs normal khana tha",
    )
    assert "BREAK THE INTERVIEW TRAP & FOLLOW NATURAL RHYTHM" in direct_prompt
    assert "React → Connect → Add Personality → Optional Follow-up" in direct_prompt
    assert "HANDLING SHORT USER REPLIES" in direct_prompt
    assert "comfort food" in direct_prompt

    final_prompt, _ = build_final_prompt(
        persona=riya,
        rag_context="",
        chat_history=[],
        user_message="bs badhiya",
    )
    assert "BREAK THE INTERVIEW TRAP & FOLLOW NATURAL RHYTHM" in final_prompt
    assert "React → Connect → Add Personality → Optional Follow-up" in final_prompt

