"""
tests/test_typing_delay.py
--------------------------
Unit tests for the realistic human-like typing delay calculator.
Verifies word count tiers, punctuation pauses, bounds, and settings integration.
"""

import pytest
from app.chat.typing_delay import calculate_typing_delay
from app.config.settings import settings


def test_empty_or_whitespace_message():
    """Empty or whitespace text should have 0 delay."""
    assert calculate_typing_delay("") == 0.0
    assert calculate_typing_delay("   \n\t  ") == 0.0
    assert calculate_typing_delay(None) == 0.0


def test_short_replies_delay_bracket():
    """1–4 words should yield approximately 0.6–1.5s delay."""
    short_reply = "Bas badhiya."
    delays = [calculate_typing_delay(short_reply) for _ in range(20)]
    for d in delays:
        assert 0.5 <= d <= 1.8, f"Delay {d} out of expected bracket for short reply"


def test_medium_replies_delay_bracket():
    """5–10 words should yield approximately 1.0–2.5s delay."""
    medium_reply = "Main bhi badhiya hu! Aaj ka din kaisa raha?"
    delays = [calculate_typing_delay(medium_reply) for _ in range(20)]
    for d in delays:
        assert 0.8 <= d <= 2.8, f"Delay {d} out of expected bracket for medium reply"


def test_long_replies_delay_bracket():
    """11–20 words should yield approximately 1.8–3.5s delay."""
    long_reply = (
        "Haha simple ghar ka khana bhi best hota hai. "
        "Waise tumhari comfort food kya hai—dal-chawal type ya kuch spicy?"
    )
    delays = [calculate_typing_delay(long_reply) for _ in range(20)]
    for d in delays:
        assert 1.5 <= d <= 4.0, f"Delay {d} out of expected bracket for long reply"


def test_maximum_delay_cap():
    """Very long messages should never exceed the 6.0s ceiling."""
    very_long_reply = (
        "Office ka workload kabhi kabhi itna badh jata hai ki lunch ka bhi time nahi milta. "
        "Lekin self-care zaroori hai. Thoda break lena chahiye, deep breathe karna chahiye, "
        "aur chai ya coffee ke saath relax karna chahiye! Tumhara kya plan hai shaam ko?"
    )
    delays = [calculate_typing_delay(very_long_reply) for _ in range(30)]
    for d in delays:
        assert 0.5 <= d <= 6.0, f"Delay {d} exceeded safety bound 6.0s"


def test_punctuation_adds_natural_pause():
    """Punctuation marks (commas, periods, question marks) add slight natural pauses."""
    plain_text = "word one two three four five six seven eight nine ten"
    punctuated_text = "word one, two... three? four! five, six. seven? eight! nine... ten?"
    
    # Over 50 samples, average delay with heavy punctuation should be higher than plain text
    plain_avg = sum(calculate_typing_delay(plain_text) for _ in range(50)) / 50.0
    punct_avg = sum(calculate_typing_delay(punctuated_text) for _ in range(50)) / 50.0
    assert punct_avg > plain_avg


def test_settings_enable_typing_delay_flag():
    """Settings flag enable_typing_delay is loaded and defaults to True."""
    assert hasattr(settings, "enable_typing_delay")
    assert settings.enable_typing_delay is True
