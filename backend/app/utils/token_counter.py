"""
utils/token_counter.py
----------------------
Lightweight token estimation utilities.

Used by:
  - prompt_builder.py  — to stay within context window limits
  - memory/conversation_memory.py — to decide when to summarise old turns

Uses tiktoken (cl100k_base encoding) as a reasonable approximation across
different model families. For production, swap in the provider-specific
tokenizer as needed.
"""

import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    _TIKTOKEN_AVAILABLE = True
except Exception:
    _TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken unavailable — falling back to character-based token estimate")


def count_tokens(text: str) -> int:
    """Return approximate token count for a string."""
    if _TIKTOKEN_AVAILABLE:
        return len(_enc.encode(text))
    # Rough fallback: ~4 chars per token
    return max(1, len(text) // 4)


def count_messages_tokens(messages: List[Dict[str, str]]) -> int:
    """Sum token counts for a list of {"role": ..., "content": ...} dicts."""
    total = 0
    for msg in messages:
        total += count_tokens(msg.get("content", ""))
        total += 4  # per-message overhead approximation
    return total


def fits_in_context(
    system_prompt: str,
    history: List[Dict[str, str]],
    user_message: str,
    max_tokens: int = 6000,
) -> bool:
    """Return True if the full prompt assembly fits within max_tokens."""
    total = (
        count_tokens(system_prompt)
        + count_messages_tokens(history)
        + count_tokens(user_message)
        + 50  # response buffer
    )
    return total <= max_tokens
