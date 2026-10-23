"""
chat/topic_picker.py
---------------------
Rule-based topic suggestion engine.

Design:
- No LLM calls — purely deterministic.
- Topics tracked per user+persona inside conversation_state JSON (no new DB tables).
- Returns a topic suggestion ONLY when the chat is slowing down.
- Stage computed from message count + elapsed days (no LLM):
    new           : 0-20 messages
    getting_close : 21-60 messages
    comfortable   : 61+  messages
- A topic is not suggested again within 20 messages of its last appearance.
- The persona also has "self-share hints" — small story seeds so the bot volunteers
  its own side of the conversation.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

_TOPICS_FILE = Path(__file__).parent / "topics.json"

try:
    with open(_TOPICS_FILE, encoding="utf-8") as _f:
        _ALL_TOPICS: List[Dict] = json.load(_f)["topics"]
except Exception as _e:
    logger.error("Failed to load topics.json: %s", _e)
    _ALL_TOPICS = []

_SLOW_CHAT_INDICATORS = {
    "hmm", "hm", "ok", "okay", "k", "ha", "haan", "ni", "nhi", "nahi",
    "accha", "achha", "ohh", "oh", "haha", "lol", "nice", "okay",
}

# How many messages must pass before the same topic can be suggested again
_TOPIC_COOLDOWN = 20


def _get_stage(message_count: int) -> str:
    if message_count <= 20:
        return "new"
    elif message_count <= 60:
        return "getting_close"
    return "comfortable"


def _is_slow_message(user_message: str) -> bool:
    """True if the user's reply is very short / low-energy."""
    clean = user_message.strip().lower().rstrip("?.! ")
    if len(clean.split()) <= 2 and clean in _SLOW_CHAT_INDICATORS:
        return True
    return len(user_message.strip().split()) <= 2


def get_topic_suggestion(
    user_message: str,
    conversation_state: Optional[Dict[str, Any]],
    message_count: int,
    persona_name: str = "",
) -> Optional[Dict[str, Any]]:
    """
    Return a topic dict (with opener_hint and self_share_hint) or None.

    Only suggests a topic when:
      1. Chat is slowing down (user short reply) AND
      2. No topic has been suggested in the last _TOPIC_COOLDOWN messages.

    Does NOT suggest if the user message is substantive (the user is already talking).
    """
    if not _is_slow_message(user_message):
        return None

    # Read state
    state = conversation_state or {}
    topics_discussed: Dict[str, int] = state.get("topics_discussed", {})
    last_topic_at: int = state.get("last_topic_suggestion_at", 0)

    # Cooldown: don't suggest if we already suggested one recently
    if message_count - last_topic_at < _TOPIC_COOLDOWN and last_topic_at > 0:
        return None

    stage = _get_stage(message_count)

    # Filter by stage and cooldown
    candidates = [
        t for t in _ALL_TOPICS
        if stage in t.get("stage", [])
        and _topic_cooldown_ok(t["id"], topics_discussed, message_count)
    ]

    if not candidates:
        # Fallback: allow any stage
        candidates = [
            t for t in _ALL_TOPICS
            if _topic_cooldown_ok(t["id"], topics_discussed, message_count)
        ]

    if not candidates:
        return None

    chosen = random.choice(candidates)
    opener = random.choice(chosen.get("opener_hints", [""]))
    self_share = random.choice(chosen.get("self_share_hints", [""]))

    return {
        "topic_id": chosen["id"],
        "category": chosen.get("category", ""),
        "opener_hint": opener,
        "self_share_hint": self_share,
        "message_count": message_count,
    }


def _topic_cooldown_ok(topic_id: str, discussed: Dict[str, int], now: int) -> bool:
    if topic_id not in discussed:
        return True
    last = discussed[topic_id]
    return now - last >= _TOPIC_COOLDOWN


def update_topics_state(
    conversation_state: Dict[str, Any],
    topic_id: Optional[str],
    message_count: int,
) -> Dict[str, Any]:
    """
    Mark a topic as discussed. Call after injecting a topic suggestion.
    Returns updated conversation_state dict (caller must persist it).
    """
    if topic_id:
        discussed = conversation_state.get("topics_discussed", {})
        discussed[topic_id] = message_count
        conversation_state["topics_discussed"] = discussed
        conversation_state["last_topic_suggestion_at"] = message_count
    return conversation_state


def get_relationship_stage(message_count: int) -> str:
    """Public helper — rule-based, no LLM call."""
    return _get_stage(message_count)


def compute_persona_mood(
    message_count: int,
    hour_of_day: int,
    day_of_week: int,
) -> str:
    """
    Rule-based persona mood — no LLM call, zero latency.
    Returns one of: happy, sleepy, busy, excited, chill, sulky
    """
    # Sleepy: late night or early morning on weekdays
    if hour_of_day >= 23 or hour_of_day < 6:
        return "sleepy"
    # Excited: Friday evening / Saturday
    if day_of_week == 4 and hour_of_day >= 17:
        return "excited"
    if day_of_week == 5:
        return "excited"
    # Busy: weekday mornings
    if day_of_week < 5 and 9 <= hour_of_day <= 12:
        return "busy"
    # Monday blues
    if day_of_week == 0:
        return "busy"
    # Default: happy for the first 20 messages, then chill
    if message_count < 20:
        return "happy"
    return "chill"
