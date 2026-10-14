"""
tests/test_prompt_and_validator.py
----------------------------------
Unit tests for modular prompt builder, language directives,
AI identity inquiry detection, and response validator.
"""

from app.chat.prompt_builder import (
    build_final_prompt, is_political_or_religious_message, is_ai_identity_inquiry
)
from app.chat.response_validator import validate_and_clean_response


def test_sensitive_deflection_and_identity_detection():
    # Political query
    assert is_political_or_religious_message("What do you think about the upcoming elections and Modi?") is True
    assert is_political_or_religious_message("Let's talk about good coffee spots in Brooklyn") is False

    # AI identity query
    assert is_ai_identity_inquiry("Wait, are you an AI or a real person?") is True
    assert is_ai_identity_inquiry("Tum real ho ya AI bot?") is True
    assert is_ai_identity_inquiry("What are you doing today?") is False


def test_modular_prompt_assembly():
    persona = {
        "id": "profile_1",
        "name": "Luna",
        "age": 26,
        "city": "Brooklyn, New York",
        "short_bio": "Photographer with dry wit",
        "personality_traits": ["observant", "warm"],
        "tone": "relaxed, warm",
        "backstory": "Luna is a photographer in Greenpoint.",
        "conversation_style_rules": ["Keep replies short."],
        "system_prompt_template": "You are Luna, 26, living in Brooklyn.",
    }

    history = [
        {"role": "user", "content": "Hey Luna"},
        {"role": "assistant", "content": "Hey 🙂 Glad you're here."},
    ]

    rag_context = "• Luna works with 35mm film."
    rolling_summary = "User and Luna previously spoke about coffee."
    conv_state = {"current_topic": "coffee", "open_threads": ["favorite brew"]}

    system_prompt, trimmed = build_final_prompt(
        persona=persona,
        rag_context=rag_context,
        chat_history=history,
        user_message="Do you shoot film?",
        rolling_summary=rolling_summary,
        conversation_state=conv_state,
    )

    # Verify modular components exist in prompt
    assert "Luna" in system_prompt
    assert "MODULE A: CORE CONVERSATIONAL POLICIES" in system_prompt
    assert "HISTORICAL CONVERSATION SUMMARY" in system_prompt
    assert "35mm film" in system_prompt
    assert "Current Topic: coffee" in system_prompt
    assert len(trimmed) == 2


def test_response_validator_cleanup():
    # 1. Robotic disclaimer stripping
    robotic_reply = "As an AI language model, I think Brooklyn is great for photography."
    cleaned = validate_and_clean_response(robotic_reply, "Luna")
    assert "as an ai language model" not in cleaned.lower()
    assert "Brooklyn is great for photography" in cleaned

    # 2. Markdown fence stripping
    fenced_reply = "```\nHey, that sounds like a fun plan!\n```"
    cleaned_fence = validate_and_clean_response(fenced_reply, "Luna")
    assert "```" not in cleaned_fence
    assert "Hey, that sounds like a fun plan!" in cleaned_fence

    # 3. Empty reply fallback
    empty_reply = "   "
    fallback = validate_and_clean_response(empty_reply, "Luna")
    assert len(fallback) > 5

    # 4. Repetition trimming
    recent = [{"role": "assistant", "content": "Hey! How is your day?"}]
    redundant_greeting = "Hey! That is so cool!"
    cleaned_rep = validate_and_clean_response(redundant_greeting, "Luna", recent_history=recent)
    assert not cleaned_rep.lower().startswith("hey!")
