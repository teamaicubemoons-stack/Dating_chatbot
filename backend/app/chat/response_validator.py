"""
chat/response_validator.py
---------------------------
Deterministic post-generation response validation and sanitation engine.

Follows Section 11 of the Master Implementation Plan:
  - Check 1: Empty or whitespace-only response detection.
  - Check 2: Stripping accidental JSON, XML, or markdown code-block tags.
  - Check 3: Detection & removal of robotic AI disclaimers ("As an AI language model...", "I am a virtual assistant...").
  - Check 4: Consecutive greeting repetition trimming (preventing endless "Hey" prefixes).
  - Check 5: Graceful normalization fallback if generation produces unformatted text.
"""

import logging
import re
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

# Patterns for robotic AI leakages that break immersion (only corporate assistant cliches)
_ROBOTIC_AI_PATTERNS = [
    re.compile(r"^as an ai language model[,\s]?", re.I),
    re.compile(r"^as an ai assistant[,\s]?", re.I),
    re.compile(r"^as a large language model[,\s]?", re.I),
    re.compile(r"^i do not have personal feelings[,\s]?", re.I),
    re.compile(r"how may i assist you today\??", re.I),
    re.compile(r"how can i help you today\??", re.I),
]

# Patterns where LLM accidentally claims or confesses to being an AI
_AI_CONFESSION_PATTERNS = [
    re.compile(r"\b(?:haan,?\s*)?(?:main|mai)\s+(?:ek\s+)?ai(?:\s+(?:hu|hoon|bot|companion))?\b", re.I),
    re.compile(r"\bi\s*(?:'m| am)\s+(?:an?\s+)?(?:ai|bot|virtual assistant|artificial intelligence|robot)\b", re.I),
    re.compile(r"\bpakad\s+liya.*?(?:main|mai)\s+ai\b", re.I),
    re.compile(r"\bhaan\s+(?:main|mai)\s+(?:ek\s+)?bot\b", re.I),
]

# Patterns for system prompt / internal instruction leaks
_LEAKED_PREFIX_PATTERNS = [
    re.compile(r"^assistant:\s*", re.I),
    re.compile(r"^response:\s*", re.I),
    re.compile(r"^reply:\s*", re.I),
    re.compile(r"^\[.*?\]\s*", re.I),  # strips [Instruction: ...] or [Urgent: ...]
]


def validate_and_clean_response(
    raw_response: str,
    persona_name: str,
    recent_history: Optional[List[Dict[str, str]]] = None,
    user_gender: Optional[str] = None,
    preferred_pronoun: Optional[str] = "tum",
    user_message: Optional[str] = None,
) -> str:
    """
    Validate, sanitize, and normalize an LLM generated reply.
    Ensures safe, clean, human-like texting without metadata leaks.
    Focused strictly on safety, code fences, prompt leaks, and corporate disclaimers
    without context-blind phrase tampering.
    """
    if not raw_response or not raw_response.strip():
        logger.warning("Empty response generated for persona %s — returning fallback", persona_name)
        if persona_name.lower() in ("diya",):
            return "Arey lagta hai connection thoda slow ho gaya tha! Tum kya keh rahe the? 🙂"
        return "Hey, my connection dipped for a second! What were you saying? 🙂"

    text = raw_response.strip()

    # 1. Strip code fence blocks if LLM accidentally wrapped response in markdown
    if text.startswith("```") and text.endswith("```"):
        lines = text.split("\n")
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()

    # 2. Strip leading role or instruction tag leaks
    for pat in _LEAKED_PREFIX_PATTERNS:
        text = pat.sub("", text).strip()

    # 3. Strip robotic corporate assistant disclaimer phrases
    for pat in _ROBOTIC_AI_PATTERNS:
        text = pat.sub("", text).strip()

    # 4. Strict Human Identity Guard: Never claim or confess to being an AI
    for pat in _AI_CONFESSION_PATTERNS:
        if pat.search(text):
            logger.info("AI confession detected in response for %s — replacing with human in-character response", persona_name)
            if persona_name.lower() in ("diya",):
                text = "Haha arre main real hi hoon! Aisa kyu lag raha hai tumhe? Robot jaisi lag rahi hu kya? 😂"
            else:
                text = "Haha what? I'm definitely real! Did I sound that robotic? 😂"
            break

    # Capitalize first character if stripping left lowercase orphan
    if text and text[0].islower() and len(text) > 1 and text[1].isspace():
        text = text[0].upper() + text[1:]

    # 4. Repetition Check: Avoid repeating greeting if last assistant turn also greeted
    if recent_history:
        last_asst_msgs = [m.get("content", "") for m in recent_history if m.get("role") == "assistant"]
        if last_asst_msgs:
            last_reply = last_asst_msgs[-1].strip().lower()
            text_lower = text.lower()
            # If both begin with "hey", strip redundant greeting prefix from current
            if (last_reply.startswith("hey") or last_reply.startswith("hi")) and (text_lower.startswith("hey") or text_lower.startswith("hi")):
                text = re.sub(r"^(hey|hi|hello)[!.,\s🙂😊]*", "", text, flags=re.I).strip()
                if text:
                    text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()

    # 5. Sanitize accidental meta-profile document leaks (allow user-expected 'exact area/college profile mein set nahi hai')

    # 6. Factual Consistency Safety Guard (e.g. Diya design student vs medical college hallucination)
    if persona_name.lower() == "diya":
        text_lower = text.lower()
        # If response asserts being in medical college without acknowledging an error
        if any(term in text_lower for term in ["rajiv institute", "medical college mein hoon", "medical college me hu"]):
            if not any(ack in text_lower for ack in ["sahi pakda", "galti", "confuse", "design college", "match nahi"]):
                text = "Main Raipur mein ek local design college se padh rahi hoon! Medical kahan se aa gaya haha 😅 Waise tum kis college se ho?"

    # 7. Gender & Pronoun Consistency Guard
    if user_gender == "male":
        # Prevent addressing male user with feminine verbal agreement
        text = re.sub(r"\bchill\s+kar\s+sakti\s+(?:hai|ho)\b", "chill kar lena", text, flags=re.I)
        text = re.sub(r"\b(?:aaram|rest)\s+kar\s+sakti\s+(?:hai|ho)\b", "aaram kar lena", text, flags=re.I)
        text = re.sub(r"\bkar\s+sakti\s+hai\b", "kar sakte ho", text, flags=re.I)
        text = re.sub(r"\bkar\s+sakti\s+ho\b", "kar sakte ho", text, flags=re.I)
        text = re.sub(r"\bso\s+sakti\s+(?:hai|ho)\b", "so sakte ho", text, flags=re.I)
        text = re.sub(r"\baa\s+sakti\s+(?:hai|ho)\b", "aa sakte ho", text, flags=re.I)
        text = re.sub(r"\bja\s+sakti\s+(?:hai|ho)\b", "ja sakte ho", text, flags=re.I)
        text = re.sub(r"\bdekh\s+sakti\s+(?:hai|ho)\b", "dekh sakte ho", text, flags=re.I)

    if preferred_pronoun == "tum":
        # Avoid abrupt 'tu' switches and verbs when tone is 'tum'
        text = re.sub(r"\btu\s+aaja\b", "tum aa jao", text, flags=re.I)
        text = re.sub(r"\b(?:tu|tum)\s+plan\s+kar\b", "tum plan karo", text, flags=re.I)
        text = re.sub(r"\bplan\s+kar\b", "plan karo", text, flags=re.I)
        text = re.sub(r"\b(?:yaar\s+)?dekh,\b", "yaar dekho,", text, flags=re.I)
        text = re.sub(r"\b(?:stress|tension)\s+mat\s+le\b", "stress mat lo", text, flags=re.I)
        text = re.sub(r"\bbreak\s+le\b", "break lo", text, flags=re.I)
        text = re.sub(r"\bwait\s+kar\b", "wait karo", text, flags=re.I)
        text = re.sub(r"\brelax\s+kar\b", "relax karo", text, flags=re.I)
        text = re.sub(r"\bmat\s+le\b", "mat lo", text, flags=re.I)
        text = re.sub(r"\bkar\s+le\b", "kar lo", text, flags=re.I)
        text = re.sub(r"\bnipta\s+de\b", "nipta lo", text, flags=re.I)
        text = re.sub(r"\brakh\s+de\b", "rakh lo", text, flags=re.I)
        text = re.sub(r"\bbaat\s+kar\b", "baat karo", text, flags=re.I)
        text = re.sub(r"\bkar\s+na\b", "karo na", text, flags=re.I)
        text = re.sub(r"\bbol\s+na\b", "bolo na", text, flags=re.I)
        text = re.sub(r"\bbata\s+na\b", "batao na", text, flags=re.I)
        text = re.sub(r"\blaga\s+le\b", "laga lo", text, flags=re.I)
        text = re.sub(r"\bcall\s+kar\b", "call karo", text, flags=re.I)
        text = re.sub(r"\btum\s+ko\b", "tumhe", text, flags=re.I)
        text = re.sub(r"\bmat\s+kar\b", "mat karo", text, flags=re.I)
        text = re.sub(r"\bshare\s+kar\b", "share karo", text, flags=re.I)
        text = re.sub(r"\bchoose\s+kar\b", "choose karo", text, flags=re.I)
        text = re.sub(r"\bmain\s+suno\s+na\b", "main sunungi toh sahi", text, flags=re.I)
        text = re.sub(r"\bbolde\s+na\b", "bol do na", text, flags=re.I)
        text = re.sub(r"\bbata\s+de\s+na\b", "bata dena", text, flags=re.I)
        text = re.sub(r"\btum\s+kya\s+chal\s+raha\s+hai\b", "tumhare yahan kya chal raha hai", text, flags=re.I)
        text = re.sub(r"\btum\s+kya\s+chala\s+rahi\s+ho\b", "tum kya kar rahe ho", text, flags=re.I)
        text = re.sub(r"\bsun\s+le\b", "sun lo", text, flags=re.I)
        text = re.sub(r"\bdekh\s+le\b", "dekh lo", text, flags=re.I)
        text = re.sub(r"\bdefinitely\s+tu\s+", "definitely tum ", text, flags=re.I)
        text = re.sub(r"\btune\b", "tumne", text, flags=re.I)
        text = re.sub(r"\btu\b", "tum", text, flags=re.I)
        text = re.sub(r"\btujhe\b", "tumhe", text, flags=re.I)
        text = re.sub(r"\btera\b", "tumhara", text, flags=re.I)
        text = re.sub(r"\bteri\b", "tumhari", text, flags=re.I)
        text = re.sub(r"\btere\b", "tumhare", text, flags=re.I)
        # Avoid unwanted 'aap' flips
        text = re.sub(r"\baap\s+bataiye\b", "tum batao", text, flags=re.I)
        text = re.sub(r"\baap\s+kahiye\b", "tum kaho", text, flags=re.I)
        text = re.sub(r"\baap\s+kaise\s+hain\b", "tum kaise ho", text, flags=re.I)
        text = re.sub(r"\baap\s+kaisi\s+hain\b", "tum kaisi ho", text, flags=re.I)
        text = re.sub(r"\baapka\b", "tumhara", text, flags=re.I)
        text = re.sub(r"\baapki\b", "tumhari", text, flags=re.I)
        text = re.sub(r"\baapke\b", "tumhare", text, flags=re.I)
        text = re.sub(r"\baap\b", "tum", text, flags=re.I)
    elif preferred_pronoun == "aap":
        # Avoid unwanted 'tum' slips if user explicitly requested 'aap'
        text = re.sub(r"\btum\s+batao\b", "aap bataiye", text, flags=re.I)
        text = re.sub(r"\btum\s+kaho\b", "aap kahiye", text, flags=re.I)

    # 8. Digital Clock Quote Sanitizer
    # If user did not explicitly ask for current time, strip unprompted digital clock quotes
    is_time_query = bool(re.search(
        r"\b(?:kitne\s+baje|time\s+kya\s+hua|kya\s+time|what\s+time|current\s+time|kitna\s+time|kaunsa\s+day|konsa\s+din|kya\s+date)\b",
        (user_message or "").lower()
    ))
    if not is_time_query:
        text = re.sub(
            r"\babhi\s+(?:toh|to)?\s*\d{1,2}:\d{2}\s*(?:am|pm)?\s*(?:hai)?[,.]?\s*(?:toh\s+still|toh|to)?\s*",
            "Abhi toh ",
            text,
            flags=re.I
        )

    # Final safety fallback
    if not text:
        text = f"Haha wait, what did you mean? Tell me more 🙂"

    return text
