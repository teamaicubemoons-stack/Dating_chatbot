"""
chat/typing_delay.py
---------------------
Calculates realistic human-like typing delays based on message length,
punctuation pauses, and natural variance.

Prevents the raw LLM generation latency from dictating user-perceived texting pace.
"""

import logging
import random
import re
from typing import Optional
from app.config.settings import settings

logger = logging.getLogger(__name__)


def calculate_typing_delay(text: str) -> float:
    """
    Calculate human-like typing delay in seconds.

    Calibrated to casual texting pace:
      1–4 words   -> 0.6–1.2 sec
      5–10 words  -> 1.0–2.0 sec
      11–20 words -> 1.8–3.0 sec
      21–35 words -> 2.5–4.5 sec
      36+ words   -> 3.5–6.0 sec

    Also accounts for punctuation pauses, natural jitter, and bounds.
    """
    if not text or not text.strip():
        return 0.0

    words = re.findall(r"\S+", text.strip())
    word_count = len(words)
    if word_count == 0:
        return 0.0

    # Calibrated base delay according to casual chat typing brackets
    if word_count <= 4:
        base_delay = random.uniform(0.6, 1.2)
    elif word_count <= 10:
        base_delay = random.uniform(1.0, 2.0)
    elif word_count <= 20:
        base_delay = random.uniform(1.8, 3.0)
    elif word_count <= 35:
        base_delay = random.uniform(2.5, 4.5)
    else:
        # Approximate 35-55 WPM pace capped at 6.0s
        wpm = random.uniform(35, 55)
        pace_delay = (word_count / wpm) * 60
        base_delay = max(3.5, min(pace_delay, 6.0))

    # Account for punctuation and natural pauses
    punctuation_pauses = (
        text.count(",") * 0.12
        + text.count(".") * 0.20
        + text.count("?") * 0.18
        + text.count("!") * 0.15
        + text.count("\n") * 0.25
    )

    # Keep very short replies quick and long replies bounded
    delay = base_delay + punctuation_pauses
    delay = max(0.6, min(delay, 6.0))

    # Small natural variation
    delay += random.uniform(-0.15, 0.25)
    final_delay = max(0.5, min(delay, 6.0))
    return round(final_delay, 2)


def calculate_read_delay(text: str) -> float:
    """
    Calculate realistic human reading time in seconds before typing indicator starts.
    Simulates the companion seeing, reading, and absorbing the user's message.

    Pacing:
      1–4 words   -> 0.8–1.3 sec
      5–12 words  -> 1.2–1.8 sec
      13+ words   -> 1.6–2.4 sec
    """
    if not text or not text.strip():
        return 0.6

    words = re.findall(r"\S+", text.strip())
    wc = len(words)
    if wc == 0:
        return 0.6

    if wc <= 4:
        delay = random.uniform(0.8, 1.3)
    elif wc <= 12:
        delay = random.uniform(1.2, 1.8)
    else:
        pace = 1.4 + wc * 0.04
        delay = min(2.4, pace + random.uniform(0.0, 0.3))

    return round(delay, 2)

