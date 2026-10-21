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
    re.compile(r"^as an ai[,\s]?", re.I),
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

    # Identity transparency is a product requirement. Never rewrite an honest
    # AI disclosure into a false claim of being human.

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
            # If consecutive assistant replies both begin with affirmative "haan/ha", strip redundant repetitive prefix
            elif re.match(r"^(?:haan|ha|han)[,\s!.]", last_reply) and re.match(r"^(?:haan|ha|han)[,\s!.]", text_lower):
                text = re.sub(r"^(?:haan|ha|han)[!.,\s🙂😊]*", "", text, flags=re.I).strip()
                if text:
                    text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()

            # Question suppression regex that matches trailing '?' followed by optional emojis / whitespace
            def _ends_with_q(s: str) -> bool:
                return bool(re.search(r"\?\s*[\U00010000-\U0010ffff\s]*$", s.strip()))

            asst_msgs = [m.get("content", "") for m in recent_history if m.get("role") == "assistant"]
            q_count_last_4 = sum(1 for m in asst_msgs[-4:] if "?" in m)
            last_had_q = bool(asst_msgs and "?" in asst_msgs[-1])

            # If last assistant reply had a question or 2+ questions in last 4 turns, prevent consecutive interrogation
            if (last_had_q or q_count_last_4 >= 2) and "?" in text:
                sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
                non_q_sentences = [s for s in sentences if not _ends_with_q(s)]
                if non_q_sentences:
                    text = " ".join(non_q_sentences).strip()
                else:
                    # Single sentence question - soften into a teasing observation or drop question format
                    if persona_name.lower().startswith("riya") and re.search(r"\b(?:coding|code|development|engineer|cs|language)\b", text, flags=re.I):
                        text = "haha pata hi tha mujhe, pakka bug dekh ke khush hone walo me se ho 😂"
                    else:
                        text = re.sub(r"^(?:kya\s+|waise\s+kya\s+)", "", text, flags=re.I)
                        text = re.sub(r"\?\s*[\U00010000-\U0010ffff\s]*$", "", text).strip()

    # Strip banned canned robotic openers ("Achha suna!", "Sahi hai!", "Hey there!", "Wow,")
    banned_opener_match = re.match(r"^(?:achha\s+suna|acha\s+suna|sahi\s+hai|hey\s+there|wow)[!.,\s🙂😊]*\s*(.+)$", text, flags=re.I | re.DOTALL)
    if banned_opener_match:
        text = banned_opener_match.group(1).strip()
        if text and text[0].islower():
            text = text[0].upper() + text[1:]

    # Strip counselor advice, stock cliches, brochure lines and mechanical tics
    text = re.sub(r"\bstress\s+toh\s+(?:aaj\s+kal\s+)?sab\s+pe\s+rehta\s+hai[.!]*\s*", "", text, flags=re.I)
    text = re.sub(r"\bthoda\s+relax\s+karne\s+ka\s+time\s+nikaal\s+lena[.!]*\s*", "", text, flags=re.I)
    text = re.sub(r"\baaram\s+se\s+baithey\s+rahe\s+na[,\s]*bina\s+kisi\s+tension\s+ke[.!]*\s*", "", text, flags=re.I)
    text = re.sub(r"\baaram\s+se\s+baithey\s+rahe\s+na[.!]*\s*", "", text, flags=re.I)
    text = re.sub(r"\bweekend\s+ka\s+best\s+part\s+toh\s+wahi\s+hota\s+hai[.!]*\s*", "", text, flags=re.I)
    text = re.sub(r"\bphir\s+kya\s+khayal\s+hai\s*\??\s*", "", text, flags=re.I)

    # Internal persona configuration must never become part of the user-facing
    # conversation. Remove only a leading meta explanation and keep its useful reply.
    text = re.sub(
        r"^(?:meri|my|the)\s+profile(?:\s+ke\s+according|['’]s\s+according|\s+mein(?:\s+mention)?|\s+me)?"
        r"\s*(?:ye\s+)?(?:mention|likha|set|configured)?\s*(?:nahi|nahin|not)?"
        r"(?:\s+hai|is)?\s*[,;:-]?\s*",
        "",
        text,
        flags=re.I,
    ).strip()
    text = re.sub(
        r"^(?:profile\s+mein|in\s+my\s+profile)\s+"
        r"(?:ye\s+)?(?:mention|likha|set|configured)\s+(?:nahi|nahin|isn't|is\s+not)"
        r"(?:\s+hai)?\s*[,;:-]?\s*",
        "",
        text,
        flags=re.I,
    ).strip()
    text = re.sub(
        r"^profile\s+(?:mein|me)\s+(?:ye\s+)?mention\s+(?:nahi|nahin)\s+(?:hai|he)?"
        r"\s*[,;:-]?\s*",
        "",
        text,
        flags=re.I,
    ).strip()

    # Brochure, LinkedIn, & mechanical tics stripping
    text = re.sub(r"\b(?:computer\s+science\s+toh\s+|cs\s+toh\s+)?bohot\s+future-oriented\s+field\s+hai[.!]*\s*", "", text, flags=re.I)
    text = re.sub(r"\bfuture-oriented\s+field\s+hai[.!]*\s*", "", text, flags=re.I)
    text = re.sub(r"\bfuture-oriented\b", "badhiya", text, flags=re.I)
    text = re.sub(r"\b(?:engineering|coding|design)\s+ka\s+scene\s+hai[.!]*\s*", "", text, flags=re.I)
    text = re.sub(r"\bka\s+scene\s+hai[.!]*\s*", "", text, flags=re.I)
    text = re.sub(r"\bscene\s+hai[.!]*\s*", "", text, flags=re.I)
    text = re.sub(r"\bkaam\s+karna\s+zaroori\s+hai[.!]*\s*", "kaam toh chalta hi rehta hai", text, flags=re.I)
    text = re.sub(r"\bcoding\s+ka\s+maza\s+toh\s+alag\s+hi\s+hota\s+hai[.!]*\s*", "", text, flags=re.I)
    text = re.sub(r"\balag\s+hi\s+hota\s+hai[.!]*\s*", "", text, flags=re.I)
    text = re.sub(r"\btumhare\s+liye\s+sabse\s+favorite\b", "tumhari favorite", text, flags=re.I)
    text = re.sub(r"\bsabse\s+favorite\b", "favorite", text, flags=re.I)
    text = re.sub(r"\bspecialization\s+kaun\s*sa\s+hai\s+tumhara\s*\??\s*", "", text, flags=re.I)

    # Strip misplaced intense love emojis in professional/casual topics
    if "😍" in text:
        work_or_tech = bool(re.search(r"\b(?:code|coding|dev|developer|engineer|job|office|kaam|work|figma|css|html)\b", (text + " " + (user_message or "")), flags=re.I))
        if work_or_tech:
            text = text.replace("😍", "")

    # Code-level Emoji frequency cap: max 1 emoji in 3-4 messages
    has_recent_assistant_emoji = False
    if recent_history:
        for m in recent_history[-3:]:
            if m.get("role") == "assistant" and re.search(r"[\U00010000-\U0010ffff]", m.get("content", "")):
                has_recent_assistant_emoji = True
                break
    # Strip weird/mismatched emojis (surfing, sunny, cupcake, sparkles)
    text = re.sub(r"[🏄☀️🧁✨]+", "", text).strip()
    if has_recent_assistant_emoji:
        text = re.sub(r"[\U00010000-\U0010ffff]", "", text).strip()
    else:
        # Cap to at most 1 emoji in current message
        emojis = re.findall(r"[\U00010000-\U0010ffff]", text)
        if len(emojis) > 1:
            for extra_e in emojis[1:]:
                text = text.replace(extra_e, "")

    # Normalize formal romanizations to casual texting Hinglish
    text = re.sub(r"\bmein\b", "me", text, flags=re.I)
    text = re.sub(r"\bkaunsa\b", "kaun sa", text, flags=re.I)
    text = re.sub(r"\bkaunsi\b", "kaun si", text, flags=re.I)
    text = re.sub(r"\bkaunse\b", "kaun se", text, flags=re.I)
    if "real hi hoon" not in text.lower():
        text = re.sub(r"\bhoon\b", "hu", text, flags=re.I)
    text = re.sub(r"\bwahaan\b", "wahan", text, flags=re.I)
    text = re.sub(r"\bkahaan\b", "kahan", text, flags=re.I)

    # Clean multiple consecutive exclamation marks
    text = re.sub(r"!{2,}", "!", text)

    # Mirror casual lowercase energy if user typed lowercase/shorthand
    if user_message and user_message.strip():
        um_clean = user_message.strip()
        if (um_clean[0].islower() or re.search(r"\b(?:kkrh|mja|bs|yrr|esa|nhi|ni)\b", um_clean, flags=re.I)) and len(text) > 1:
            if not text.startswith("I "):
                text = text[0].lower() + text[1:]

    text = re.sub(r"\s{2,}", " ", text).strip()

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
        text = re.sub(r"\btum\s+kaisi\s+ho\b", "tum kaise ho", text, flags=re.I)
    elif user_gender == "female":
        text = re.sub(r"\btum\s+kaise\s+ho\b", "tum kaisi ho", text, flags=re.I)

    if preferred_pronoun == "tum":
        # Specific pronoun-verb agreement with 'tum' (standard 'ho', NEVER 'hai')
        # Clean bilingual repeated greeting questions (e.g. 'Kaise ho? What's up?' or 'What's up? Kaise ho?')
        text = re.sub(r"(?:,\s*)?(?:kaise\s+ho|kaisa\s+hai|kaisi\s+ho)\s*[?.,!]*\s*(?:[\U00010000-\U0010ffff\s]*)\b(?:what'?s\s+up|how\s+are\s+you|how\s+is\s+it\s+going)\s*\?", " kaise ho?", text, flags=re.I)
        text = re.sub(r"\b(?:what'?s\s+up|how\s+are\s+you|how\s+is\s+it\s+going)\s*[?.,!]*\s*(?:[\U00010000-\U0010ffff\s]*)(?:kaise\s+ho|kaisa\s+hai|kaisi\s+ho)\s*\?", "Kaise ho?", text, flags=re.I)

        text = re.sub(r"\btum\s+kaise\s+ho\s+feel\s+kar\s+(?:rahe|rahi|raha)\s+ho\b", "tum kaisa feel kar rahe ho", text, flags=re.I)
        text = re.sub(r"\bkaise\s+ho\s+feel\b", "kaisa feel", text, flags=re.I)
        text = re.sub(r"\b(?:tu|tum)[,\s]+\s*kaisa\s+(?:hai|h)\b", "tum kaise ho", text, flags=re.I)
        text = re.sub(r"\b(?:tu|tum)[,\s]+\s*kaisi\s+(?:hai|h)\b", "tum kaisi ho", text, flags=re.I)
        text = re.sub(r"\b(?:tu|tum)[,\s]+\s*kaise\s+(?:hai|h)\b", "tum kaise ho", text, flags=re.I)
        text = re.sub(r"\b(?:tu|tum)\s+kaisa\b(?!\s+(?:tha|thi|the|lag|dik|feel|chal|gaya|gayi|gaye|raha|rahi|rahe|soch|experience))", "tum kaise ho", text, flags=re.I)
        text = re.sub(r"\b(aur|yaar)[,\s]+\s*kaisa\s+(?:hai|h)\b", r"\1 kaise ho", text, flags=re.I)
        text = re.sub(r"(?:^|[.!?\n]\s*)kaisa\s+(?:hai|h)\s*\?", " kaise ho?", text, flags=re.I)

        # Verbs & state agreement with 'tum'
        text = re.sub(r"\b(?:tu|tum)\s+(?:kya\s+)?kar\s+raha\s+(?:hai|h)\b", "tum kya kar rahe ho", text, flags=re.I)
        text = re.sub(r"\b(?:tu|tum)\s+(?:kya\s+)?kar\s+rahi\s+(?:hai|h)\b", "tum kya kar rahi ho", text, flags=re.I)
        text = re.sub(r"\b(?:tu|tum)\s+(?:kya\s+)?kar\s+rahe\s+(?:hai|h)\b", "tum kya kar rahe ho", text, flags=re.I)
        text = re.sub(r"\b(?:tu|tum)\s+kar\s+raha\s+(?:hai|h)\b", "tum kar rahe ho", text, flags=re.I)
        text = re.sub(r"\b(?:tu|tum)\s+kar\s+rahi\s+(?:hai|h)\b", "tum kar rahi ho", text, flags=re.I)
        text = re.sub(r"\b(?:tu|tum)\s+kahan\s+(?:hai|h)\b", "tum kahan ho", text, flags=re.I)
        text = re.sub(r"\b(?:tu|tum)\s+kahan\s+rehta\s+(?:hai|h)\b", "tum kahan rehte ho", text, flags=re.I)
        text = re.sub(r"\b(?:tu|tum)\s+kahan\s+rehti\s+(?:hai|h)\b", "tum kahan rehti ho", text, flags=re.I)
        text = re.sub(r"\b(?:tu|tum)\s+theek\s+(?:hai|h)\b", "tum theek ho", text, flags=re.I)
        text = re.sub(r"\b(?:tu|tum)\s+thik\s+(?:hai|h)\b", "tum thik ho", text, flags=re.I)
        text = re.sub(r"\b(?:tu|tum)\s+busy\s+(?:hai|h)\b", "tum busy ho", text, flags=re.I)
        text = re.sub(r"\b(?:tu|tum)\s+free\s+(?:hai|h)\b", "tum free ho", text, flags=re.I)
        text = re.sub(r"\b(?:tu|tum)\s+online\s+(?:hai|h)\b", "tum online ho", text, flags=re.I)
        text = re.sub(r"\b(?:tu|tum)\s+ready\s+(?:hai|h)\b", "tum ready ho", text, flags=re.I)

        # Habitual verbs with 'tum': 'tum bolta hai' -> 'tum bolte ho'
        text = re.sub(r"\btum\s+(?:kya\s+)?(kar|bol|soch|jaan|reh|sun|samajh|dekh|padh|chah|bhej|khel)ta\s+(?:hai|h)\b", r"tum \1te ho", text, flags=re.I)
        text = re.sub(r"\btum\s+(?:kya\s+)?(kar|bol|soch|jaan|reh|sun|samajh|dekh|padh|chah|bhej|khel)ti\s+(?:hai|h)\b", r"tum \1ti ho", text, flags=re.I)
        text = re.sub(r"\btum\s+(?:kya\s+)?(kar|bol|soch|jaan|reh|sun|samajh|dekh|padh|chah|bhej|khel)te\s+hai(?:n)?\b", r"tum \1te ho", text, flags=re.I)

        # Short imperatives: 'tum bata' -> 'tum batao', 'tum bol' -> 'tum bolo'
        text = re.sub(r"\btum\s+bata\b(?!\s*(?:o|kar|ke|te|ti|ta|dena))", "tum batao", text, flags=re.I)
        text = re.sub(r"\btum\s+bol\b(?!\s*(?:o|kar|ke|te|ti|ta|dena))", "tum bolo", text, flags=re.I)
        text = re.sub(r"\btum\s+sun\b(?!\s*(?:o|kar|ke|te|ti|ta|dena))", "tum suno", text, flags=re.I)
        text = re.sub(r"\btum\s+dekh\b(?!\s*(?:o|kar|ke|te|ti|ta|dena))", "tum dekho", text, flags=re.I)

        # Avoid abrupt 'tu' switches and verbs when tone is 'tum'
        text = re.sub(r"\btu\s+aaja\b", "tum aa jao", text, flags=re.I)
        text = re.sub(r"\b(?:tu|tum)\s+plan\s+kar\b", "tum plan karo", text, flags=re.I)
        text = re.sub(r"\bplan\s+kar\b", "plan karo", text, flags=re.I)
        text = re.sub(r"\bbreak(?:-break)?\s+(?:lo|le)\s+lena\b", "break le lena", text, flags=re.I)
        text = re.sub(r"\bbreak\s+le\b(?!\s*lena)", "break lo", text, flags=re.I)
        text = re.sub(r"\bdekh\s*,", "dekho,", text, flags=re.I)
        text = re.sub(r"\b(?:aur\s+|yaar\s+)?dekh\b(?!\s*(?:te|ti|ta|enge|kar))", "dekho", text, flags=re.I)
        text = re.sub(r"\bkaam\s+nahi\s+sunta\b", "kaam chain nahi lene deta", text, flags=re.I)
        text = re.sub(r"\b(?:stress|tension)\s+mat\s+le\b", "stress mat lo", text, flags=re.I)
        text = re.sub(r"\bwait\s+kar\b", "wait karo", text, flags=re.I)
        text = re.sub(r"\brelax\s+kar\b", "relax karo", text, flags=re.I)
        text = re.sub(r"\bchill\s+kar\b(?!\s*(?:o|lena|liya|ke|te|ti|ta))", "chill karo", text, flags=re.I)
        text = re.sub(r"\benjoy\s+kar\b(?!\s*(?:o|lena|liya|ke|te|ti|ta))", "enjoy karo", text, flags=re.I)
        text = re.sub(r"\btry\s+kar\b(?!\s*(?:o|lena|liya|ke|te|ti|ta))", "try karo", text, flags=re.I)
        text = re.sub(r"\bcheck\s+kar\b(?!\s*(?:o|lena|liya|ke|te|ti|ta))", "check karo", text, flags=re.I)
        text = re.sub(r"\bstart\s+kar\b(?!\s*(?:o|lena|liya|ke|te|ti|ta))", "start karo", text, flags=re.I)
        text = re.sub(r"\bhelp\s+kar\b(?!\s*(?:o|lena|liya|ke|te|ti|ta))", "help karo", text, flags=re.I)
        text = re.sub(r"\btext\s+kar\b(?!\s*(?:o|lena|liya|ke|te|ti|ta))", "text karo", text, flags=re.I)
        text = re.sub(r"\bmessage\s+kar\b(?!\s*(?:o|lena|liya|ke|te|ti|ta))", "message karo", text, flags=re.I)
        text = re.sub(r"\bchinta\s+mat\s+kar\b", "chinta mat karo", text, flags=re.I)
        text = re.sub(r"\bload\s+mat\s+le\b", "load mat lo", text, flags=re.I)
        text = re.sub(r"\btum\s+se\s+janne\s+k[ai]\s+curiosity\b", "tumhe janne ki curiosity", text, flags=re.I)
        text = re.sub(r"\btum\s+se\s+janne\b", "tumhe janne", text, flags=re.I)
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

    # 8. Hinglish Verb Morphology & Natural Phrasing Guards
    # A. Continuous aspect with bare root stem (NEVER imperative -o ending with raha/rahi/rahe)
    text = re.sub(r"\bsuno\s+(rah[aie]+)\b", r"sun \1", text, flags=re.I)
    text = re.sub(r"\bdekho\s+(rah[aie]+)\b", r"dekh \1", text, flags=re.I)
    text = re.sub(r"\bbolo\s+(rah[aie]+)\b", r"bol \1", text, flags=re.I)
    text = re.sub(r"\bkaro\s+(rah[aie]+)\b", r"kar \1", text, flags=re.I)
    text = re.sub(r"\bsocho\s+(rah[aie]+)\b", r"soch \1", text, flags=re.I)
    text = re.sub(r"\bpadho\s+(rah[aie]+)\b", r"padh \1", text, flags=re.I)
    text = re.sub(r"\bkhelo\s+(rah[aie]+)\b", r"khel \1", text, flags=re.I)
    text = re.sub(r"\bbatao\s+(rah[aie]+)\b", r"bata \1", text, flags=re.I)
    text = re.sub(r"\bsamjho\s+(rah[aie]+)\b", r"samajh \1", text, flags=re.I)
    text = re.sub(r"\broko\s+(rah[aie]+)\b", r"rok \1", text, flags=re.I)
    text = re.sub(r"\bchalo\s+(rah[aie]+)\b", r"chal \1", text, flags=re.I)
    text = re.sub(r"\baao\s+(rah[aie]+)\b", r"aa \1", text, flags=re.I)
    text = re.sub(r"\bjaao\s+(rah[aie]+)\b", r"jaa \1", text, flags=re.I)

    # B. Broken compound verbs & awkward dialect forms (e.g. 'break lo liyo' -> 'break le lena')
    text = re.sub(r"\bbreak\s+lo\s+lo\b", "break le lo", text, flags=re.I)
    text = re.sub(r"\b(dekh|sun|bol|kar|soch|padh)o\s+ke\b", r"\1 ke", text, flags=re.I)
    text = re.sub(r"\bbreak\s+(?:lo|le)\s+(?:liyo|le)\b", "break le lena", text, flags=re.I)
    text = re.sub(r"\bbreak\s+lo\s+lena\b", "break le lena", text, flags=re.I)
    text = re.sub(r"\blo\s+liyo\b", "le lena", text, flags=re.I)
    text = re.sub(r"\ble\s+liyo\b", "le lena", text, flags=re.I)
    text = re.sub(r"\bkar\s+liyo\b", "kar lena", text, flags=re.I)
    text = re.sub(r"\bde\s+liyo\b", "de dena", text, flags=re.I)
    text = re.sub(r"\bdekh\s+liyo\b", "dekh lena", text, flags=re.I)
    text = re.sub(r"\bsun\s+liyo\b", "sun lena", text, flags=re.I)
    text = re.sub(r"\bbol\s+liyo\b", "bol lena", text, flags=re.I)
    text = re.sub(r"\bpadh\s+liyo\b", "padh lena", text, flags=re.I)
    text = re.sub(r"\bso\s+liyo\b", "so lena", text, flags=re.I)
    text = re.sub(r"\brakh\s+liyo\b", "rakh lena", text, flags=re.I)
    text = re.sub(r"\bnipta\s+liyo\b", "nipta lena", text, flags=re.I)
    text = re.sub(r"\baa\s+liyo\b", "aa jana", text, flags=re.I)

    # C. Awkward 'too much' phrasing in Hinglish advice
    text = re.sub(r"\b(?:tum\s+)?(?:stress|tension)\s+mat\s+lo\s+too\s+much\b", "zyada stress mat lo", text, flags=re.I)
    text = re.sub(r"\btoo\s+much\s+(?:stress|tension)\s+mat\s+lo\b", "zyada stress mat lo", text, flags=re.I)
    text = re.sub(r"\bmat\s+lo\s+too\s+much\b", "zyada mat lo", text, flags=re.I)

    # 9. Digital Clock Quote Sanitizer
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

    # 10. Venue & Geographic Hallucination Guard
    # If companion claims familiarity with an unverified ground or venue (e.g. 'BTI ground mein accha vibe hota hai usually')
    if persona_name.lower() in ("riya", "riya kapoor"):
        if "bti ground" in text.lower() and ("accha vibe" in text.lower() or "usually" in text.lower()):
            text = "BTI ground? Yeh kaunse city ya area mein hai waise? Main Delhi/Bangalore se hoon toh mujhe exact idea nahi hai haha 😅"
        text = re.sub(
            r"\b([A-Za-z0-9]+(?:\s+[A-Za-z0-9]+)?\s+ground)\s+mein\s+ac?ch?a\s+vibe\s+hota\s+hai(?:\s+usually)?\b[^\n.!?]*[.,!?]?",
            r"\1? Yeh kaunse city ya area mein hai waise? Main Delhi/Bangalore se hoon toh mujhe exact idea nahi hai haha 😅",
            text,
            flags=re.I
        )

    # Sanitize breaking-character confessions like 'honestly maine assume kar liya'
    text = re.sub(
        r"\b(?:honestly\s+)?maine\s+assume\s+kar\s+liya(?:,\s*mujhe\s+exact\s+nahi\s+pata)?\b",
        "mujhe laga koi popular local spot hoga",
        text,
        flags=re.I
    )

    # 11. Festival Context Sanitizer
    if user_message and "navratri" in user_message.lower():
        text = re.sub(r"\bnavratri\s+aane\s+wali\s+hai\b", "Navratri ka vibe toh chal hi raha hai", text, flags=re.I)
        text = re.sub(r"\bexact\s+dates(?:\s+ya\s+events)?\s+maine\s+abhi\s+check\s+nahi\s+kiye(?:\s+honestly)?\b", "vibe kaafi accha chal raha hai", text, flags=re.I)

    # Final safety fallback
    if not text:
        text = f"Haha wait, what did you mean? Tell me more 🙂"

    return text


def arrange_and_format_response(
    raw_response: str,
    persona_name: str,
    recent_history: Optional[List[Dict[str, str]]] = None,
    user_gender: Optional[str] = None,
    preferred_pronoun: Optional[str] = "tum",
    user_message: Optional[str] = None,
) -> str:
    """
    Arranges, styles, and formats the raw LLM response from OpenAI according to:
      1. User-specific context (preferred pronoun 'tum'/'aap', grammatical gender agreement).
      2. Persona character & natural mobile texting format (brevity, casing, emoji moderation).
      3. Clean-up of AI leaks, code fences, corporate assistant disclaimers, or metadata tags.
      4. Arranging multi-line / verbose responses into clean conversational chat format.
    """
    # 1. Base validation and cleaning with user context
    text = validate_and_clean_response(
        raw_response=raw_response,
        persona_name=persona_name,
        recent_history=recent_history,
        user_gender=user_gender,
        preferred_pronoun=preferred_pronoun,
        user_message=user_message,
    )

    # 2. Arrange multi-paragraph or multi-line responses into clean chat bubbles
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if len(paragraphs) > 1:
        text = " ".join(paragraphs)

    # 3. Clean redundant spaces and punctuation combos
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s*([.,!?])", r"\1", text)
    text = re.sub(r"\?{2,}", "?", text)
    text = re.sub(r"!{2,}", "!", text)

    return text
