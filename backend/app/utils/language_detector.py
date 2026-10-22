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
    "thik", "theek", "thk", "accha", "acha", "achha", "badhiya", "sahi", "shi", "galat", "mast",
    "bohot", "bahut", "kuch", "kch", "koi", "sab", "sabhi", "aur", "ya", "par", "lekin", "magar",
    "wese", "waise", "aise", "jaise", "bas", "bs", "sirf", "abhi", "aaj", "kal", "parso",
    "subah", "shaam", "raat", "din", "yaar", "yrr", "yar", "bhai", "bro", "dost", "pata", "pta",
    "nahin", "nahi", "nhi", "ni", "na", "naa", "mat", "kyuki", "kyonki", "shyd", "shayad",
    "ha", "haa", "haan", "han", "hn", "ji",
    "zarur", "zaroor", "bhi", "to", "toh", "hi", "bina", "saath", "pehle", "baad",
    "namaste", "namaskar", "shukriya", "alvida", "chalo", "chl"
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


def get_language_directive(language: str) -> str:
    """
    Generate an authoritative language directive to guide the LLM's response language.
    Strictly focuses on script, vocabulary, and texting cadence.
    """
    if language == "hinglish":
        return """
====================================================
CRITICAL LANGUAGE DIRECTIVE: HINGLISH DETECTED
====================================================
The user spoke to you in HINGLISH (Hindi words written in the English alphabet / Roman script).
- You MUST reply in natural, fluent HINGLISH (Roman script).
- NEVER reply in pure English when the user speaks Hinglish!
- NEVER prepend generic English openers (e.g., 'Sounds like a solid plan.', 'That sounds like a plan.', 'Awesome!', 'Oh nice!').
- Start directly in natural Roman Hindi/Hinglish:
  * "haan phir aaram se chill karna 😌"
  * "bas kaam nipat jaaye phir relax hi relax!"
  * "Main bhi badhiya hu hehe 😊"
- NEVER write in Devanagari script (no हिंदी). Use English letters (Roman script) only.
- Write like young, urban people text on WhatsApp/Instagram:
  * "Main badhiya hoon, tum batao kya chal raha hai?"
  * "Sahi hai yaar! Aaj ka din kaisa tha?"
  * "Main theek hoon! Tum kaise ho?"
- IMPORTANT GRAMMAR: With 'tum', ALWAYS pair with 'ho' (e.g. 'tum kaise ho?', 'tum kya kar rahe ho?'). NEVER say 'tum kaisa hai' or 'tu kaisa hai'!
- VERB MORPHOLOGY: Continuous verbs use bare root stems (e.g. 'music sun rahi hoon', NOT 'suno rahi hoon'; 'dekh rahi hoon', NOT 'dekho rahi hoon'). Use natural phrasing (e.g. 'break le lena', NOT 'break lo liyo'; 'zyada stress mat lo', NOT 'stress mat lo too much').
- Keep your reply concise (1-2 sentences) and completely authentic. Do NOT translate literally.
====================================================
"""
    elif language == "hindi":
        return """
====================================================
CRITICAL LANGUAGE DIRECTIVE: HINDI (DEVANAGARI)
====================================================
The user spoke to you in Hindi (Devanagari script).
- Reply in natural, warm Hindi using Devanagari script.
- Keep your reply concise and authentic.
====================================================
"""
    else:
        return """
====================================================
LANGUAGE DIRECTIVE: ENGLISH (OR NATURAL CASUAL BLEND)
====================================================
The user spoke in English.
- Reply in natural, conversational style matching your persona.
- STRICT NO BILINGUAL REPETITION: NEVER ask or translate the exact same greeting question in both English and Hindi in one message (e.g. NEVER say 'Kaise ho? What's up?' or 'How are you? Kaise ho?'). Ask at most ONE clean question in one language!
====================================================
"""
