"""
tests/test_memory_extraction.py
-------------------------------
Unit tests for candidate memory extraction, Hinglish parsing,
deduplication, and state cue detection.
"""

from app.memory.memory_extractor import extract_candidate_memories


def test_english_fact_and_preference_extraction():
    # Name extraction
    cands, emo, topic = extract_candidate_memories("My name is Rohit", profile_id="profile_1")
    assert len(cands) >= 1
    assert any("Rohit" in c.memory_text for c in cands)

    # Occupation extraction
    cands, emo, topic = extract_candidate_memories("I work as a software architect at Google", profile_id="profile_1")
    assert len(cands) >= 1
    assert any("software architect" in c.memory_text.lower() for c in cands)
    assert topic is not None

    # Preference extraction
    cands, emo, topic = extract_candidate_memories("I really love artisan sourdough bread", profile_id="profile_1")
    assert len(cands) >= 1
    assert any("sourdough" in c.memory_text.lower() for c in cands)


def test_hinglish_fact_extraction():
    # Hinglish Name
    cands, emo, topic = extract_candidate_memories("Mera naam Priya hai", profile_id="profile_1")
    assert len(cands) >= 1
    assert any("Priya" in c.memory_text for c in cands)

    # Hinglish Location
    cands, emo, topic = extract_candidate_memories("Main Mumbai me rehti hu", profile_id="profile_1")
    assert len(cands) >= 1
    assert any("Mumbai" in c.memory_text for c in cands)

    # Hinglish Preference
    cands, emo, topic = extract_candidate_memories("Mujhe filter coffee bohot pasand hai", profile_id="profile_1")
    assert len(cands) >= 1
    assert any("filter coffee" in c.memory_text.lower() for c in cands)


def test_episodic_event_and_emotion_detection():
    # Event + emotion
    msg = "I have an interview tomorrow and I feel so nervous"
    cands, emo, topic = extract_candidate_memories(msg, profile_id="profile_1")
    assert any("interview" in c.memory_text.lower() for c in cands)
    assert emo == "nervous"
    assert "interview" in (topic or "").lower()

    # Hinglish event + emotion
    msg_hi = "Kal mera exam hai thoda tension ho raha hai"
    cands, emo, topic = extract_candidate_memories(msg_hi, profile_id="profile_1")
    assert emo == "tension"
