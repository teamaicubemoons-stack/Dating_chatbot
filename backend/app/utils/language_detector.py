"""
utils/language_detector.py
---------------------------
Detects user language (English, Hinglish / Roman Hindi, or Devanagari Hindi)
and generates strict language mirroring directives for the LLM prompt.
"""

import re
from typing import Dict, Any

# Common Hinglish / Roman Hindi particles, pronouns, verbs, and conversational words
HINGLISH_VOCABULARY = {
    # Question words & Pronouns
    "kya", "kaise", "kaisa", "kaisi", "kyu", "kyun", "kaha", "kahan", "kab",
    "kitna", "kitne", "kitni", "kaun", "kon", "kisko", "kisne", "kiska", "kiski",
    "tum", "tumhe", "tujhe", "tu", "tera", "teri", "tere", "tumhara", "tumhari", "tumhare",
    "aap", "aapka", "aapki", "aapke", "main", "mai", "mera", "meri", "mere",
    "mujhe", "mujhko", "hum", "hume", "hamara", "hamari", "hamare", "wo", "woh",
    "uska", "uski", "uske", "use", "usko", "ye", "yeh", "iska", "iski", "iske", "ise", "unka",

    # Verbs & auxiliaries
    "hai", "hain", "ho", "hu", "hoon", "tha", "thi", "the", "hoga", "hogi", "honge",
    "kar", "karo", "kare", "karega", "karegi", "karenge", "karna", "karte", "karta", "karti",
    "kr", "krte", "krta", "krti", "kiya", "raha", "rahi", "rahe", "rha", "rhi", "rhe",
    "dekh", "dekho", "dekha", "bol", "bolo", "bola", "boli", "bata", "batao", "bataya",
    "sun", "suno", "suna", "samajh", "samjha", "samjhi", "chal", "chalo", "chala",
    "aao", "aaya", "aayi", "ja", "jao", "gaya", "gayi", "gaye", "le", "lo", "liya",
    "de", "do", "diya", "khana", "khaya", "soch", "socha", "mil", "mila", "mile",
    "lag", "laga", "lagi", "lage", "lagta", "lagti", "rakh", "rakho", "rakha",

    # Common modifiers, particles, greetings & slang
    "thik", "theek", "accha", "acha", "achha", "badhiya", "sahi", "galat", "mast",
    "bohot", "bahut", "kuch", "koi", "sab", "sabhi", "aur", "ya", "par", "lekin", "magar",
    "wese", "waise", "aise", "jaise", "bas", "sirf", "abhi", "aaj", "kal", "parso",
    "subah", "shaam", "raat", "din", "yaar", "bhai", "bro", "dost", "pata", "pta",
    "nahin", "nahi", "nhi", "na", "mat", "kyuki", "kyonki", "shyd", "shayad",
    "zarur", "zaroor", "bhi", "to", "toh", "hi", "bina", "saath", "pehle", "baad",
    "namaste", "namaskar", "shukriya", "alvida", "chalo"
}


def detect_language(text: str) -> str:
    """
    Classify input text as 'hindi' (Devanagari), 'hinglish' (Roman script), or 'english'.
    """
    if not text or not text.strip():
        return "english"

    # 1. Check for Devanagari Unicode block (\u0900 - \u097F)
    if re.search(r"[\u0900-\u097F]", text):
        return "hindi"

    # 2. Check for Hinglish / Roman Hindi vocabulary
    words = set(re.findall(r"[a-zA-Z]+", text.lower()))
    hinglish_matches = words.intersection(HINGLISH_VOCABULARY)

    if len(hinglish_matches) >= 1:
        return "hinglish"

    return "english"


def get_language_directive(language: str, persona: Dict[str, Any]) -> str:
    """
    Generate an authoritative language directive to guide the LLM's response language.
    """
    name = persona.get("name", "Companion")
    persona_id = persona.get("id", "")

    # Gender pronoun clue for natural verb conjugations (karta hoon vs karti hoon)
    is_female = persona_id in ("profile_1", "profile_3") or name.lower() in ("luna", "zara", "maya")

    gender_hint = (
        "Since you are female, use natural female verb endings in Hindi/Hinglish (e.g., 'main karti hoon', 'dekh rahi hoon', 'lagti hoon')."
        if is_female
        else "Use natural male verb endings in Hindi/Hinglish (e.g., 'main karta hoon', 'dekh raha hoon', 'lagta hoon')."
    )

    if language == "hinglish":
        return f"""
====================================================
CRITICAL LANGUAGE DIRECTIVE: HINGLISH DETECTED
====================================================
The user spoke to you in HINGLISH (Hindi words written in the English alphabet / Roman script).
- You MUST reply in natural, fluent HINGLISH (Roman script).
- NEVER reply in pure English when the user speaks Hinglish!
- NEVER write in Devanagari script (no हिंदी). Use English letters (Roman script) only.
- Write like young, urban people text on WhatsApp/Instagram:
  * "Main badhiya hoon, tu bata kya chal raha hai?"
  * "Job karti hoon. Yahan SF mein ek music platform bana rahi hoon. Thoda chaotic hai but I love it."
  * "Sahi hai yaar! Aaj ka din kaisa tha?"
- {gender_hint}
- Keep your reply concise (1-2 sentences) and completely authentic. Do NOT translate literally.
====================================================
"""
    elif language == "hindi":
        return f"""
====================================================
CRITICAL LANGUAGE DIRECTIVE: HINDI (DEVANAGARI)
====================================================
The user spoke to you in Hindi (Devanagari script).
- Reply in natural, warm Hindi using Devanagari script.
- {gender_hint}
- Keep your reply concise and authentic.
====================================================
"""
    else:
        return """
====================================================
LANGUAGE DIRECTIVE: ENGLISH
====================================================
The user spoke in English. Reply in natural, conversational English matching your persona.
====================================================
"""
