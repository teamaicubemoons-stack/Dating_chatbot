"""
tests/test_topic_picker.py
--------------------------
Unit tests for the rule-based topic suggestion engine:
  - Slow chat detection
  - Stage-aware candidate filtering (new, getting_close, comfortable)
  - Cooldown enforcement (no repeats within 20 messages)
  - State tracking updates
"""

from app.chat.topic_picker import (
    _is_slow_message,
    _get_stage,
    _topic_cooldown_ok,
    get_topic_suggestion,
    update_topics_state,
    _ALL_TOPICS,
)


def test_slow_chat_detection():
    assert _is_slow_message("ok") is True
    assert _is_slow_message("hmm") is True
    assert _is_slow_message("ha") is True
    assert _is_slow_message("accha") is True
    assert _is_slow_message("kuch nahi") is True
    # Substantive messages should not trigger topic suggestions
    assert _is_slow_message("Today I went to the park and met an old friend from college") is False


def test_stages():
    assert _get_stage(5) == "new"
    assert _get_stage(20) == "new"
    assert _get_stage(21) == "getting_close"
    assert _get_stage(60) == "getting_close"
    assert _get_stage(61) == "comfortable"


def test_cooldown():
    discussed = {"t_street_food": 10}
    # Turn 15 is only 5 messages later -> not ok
    assert _topic_cooldown_ok("t_street_food", discussed, 15) is False
    # Turn 31 is 21 messages later -> ok
    assert _topic_cooldown_ok("t_street_food", discussed, 31) is True
    # Undiscussed topic -> ok
    assert _topic_cooldown_ok("t_indie_music", discussed, 15) is True


def test_topic_suggestion_on_slow_message():
    state = {"topics_discussed": {}, "last_topic_suggestion_at": 0}
    suggestion = get_topic_suggestion(
        user_message="hmm",
        conversation_state=state,
        message_count=5,
        persona_name="Diya",
    )
    assert suggestion is not None
    assert "topic_id" in suggestion
    assert "opener_hint" in suggestion
    assert "self_share_hint" in suggestion


def test_no_topic_suggestion_on_active_chat():
    state = {"topics_discussed": {}, "last_topic_suggestion_at": 0}
    suggestion = get_topic_suggestion(
        user_message="I'm planning to adopt a golden retriever puppy this weekend!",
        conversation_state=state,
        message_count=5,
        persona_name="Luna",
    )
    assert suggestion is None


def test_state_update():
    state = {}
    updated = update_topics_state(state, "t_chai_coffee", 10)
    assert updated["topics_discussed"]["t_chai_coffee"] == 10
    assert updated["last_topic_suggestion_at"] == 10


if __name__ == "__main__":
    test_slow_chat_detection()
    test_stages()
    test_cooldown()
    test_topic_suggestion_on_slow_message()
    test_no_topic_suggestion_on_active_chat()
    test_state_update()
    print("[PASS] All topic_picker tests passed!")
