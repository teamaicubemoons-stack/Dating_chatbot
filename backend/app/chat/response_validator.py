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
from typing import Optional, List, Dict, Any

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

# Patterns for leaked role tags or instruction wrappers
_LEAKED_PREFIX_PATTERNS = [
    re.compile(r"^(?:assistant|diya|luna|ethan|zara|bot|system):\s*", re.I),
    re.compile(r"^\[(?:response|reply|assistant|message)\]:\s*", re.I),
]

# Patterns where LLM accidentally claims or confesses to being an AI or digital bot
_AI_CONFESSION_PATTERNS = [
    re.compile(r"\b(?:haan,?\s*)?(?:main|mai)\s+(?:ek\s+)?ai(?:\s+(?:hu|hoon|bot|companion|model))?\b", re.I),
    re.compile(r"\bi\s*(?:'m| am)\s+(?:an?\s+)?(?:ai|bot|virtual assistant|artificial intelligence|robot|digital companion|fictional companion)\b", re.I),
    re.compile(r"\bpakad\s+liya.*?(?:main|mai)\s+ai\b", re.I),
    re.compile(r"\bhaan\s+(?:main|mai)\s+(?:ek\s+)?bot\b", re.I),
    re.compile(r"\bmain\s+(?:ek\s+)?bot\s+hu\b", re.I),
    re.compile(r"\b(?:digital\s+companion|fictional\s+companion|ai\s+digital\s+companion)\b", re.I),
    re.compile(r"\bspark\s+app\s+par\s+designed\b", re.I),
    re.compile(r"\bdesigned\s+on\s+spark\b", re.I),
    re.compile(r"\blanguage\s+model\b", re.I),
    re.compile(r"\bvirtual\s+companion\b", re.I),
]

# Patterns for leaked chatbot architecture, app setup limitations, or static idling
_APP_SETUP_LEAK_PATTERNS = [
    re.compile(r"\bmain\s+(?:khud\s+se\s+)?(?:message|msg|text|ping)\s+(?:start|shuru)\s+nahi\s+kar\s+sakt[ia]\b", re.I),
    re.compile(r"\btum\s+jab\s+likhte\s+ho\s+tabhi\s+(?:main\s+)?reply\b", re.I),
    re.compile(r"\bapp\s+ka\s+setup\s+(?:hi\s+)?aisa\s+hai\b", re.I),
    re.compile(r"\bapne\s+(?:tum|aap|se)\s+ping\s+nahi\s+kar\s+sakt[ia]\b", re.I),
    re.compile(r"\bmain\s+apne\s+se\s+ping\s+nahi\b", re.I),
    re.compile(r"\bmain\s+(?:bhi\s+)?yahin\s+rahung[ia]\b", re.I),
]

# Banned try-hard metaphor punchlines (ban metaphor-style punchlines such as "trailer mode", "negotiation", "budget", "vibe hijack", "araam mode", "scene on hai")
_BANNED_METAPHOR_PATTERNS = [
    re.compile(r"\btrailer\s+mode\b", re.I),
    re.compile(r"\bpicture\s+baaki\s+hai\b", re.I),
    re.compile(r"\bneend\s+se\s+negotiation\b", re.I),
    re.compile(r"\bnegotiation\b", re.I),
    re.compile(r"\b(?:reply|shabdo?n?)\s+ka\s+budget\b", re.I),
    re.compile(r"\bbudget\s+thoda\s+limited\b", re.I),
    re.compile(r"\bbudget\s+limited\b", re.I),
    re.compile(r"\bvibe\s+hijack\b", re.I),
    re.compile(r"\b(?:araam|aram)\s+mode\b", re.I),
    re.compile(r"\bscene\s+on\s+hai\b", re.I),
    re.compile(r"\bscene\s+on\b", re.I),
    re.compile(r"\bbattery\s+low\b", re.I),
    re.compile(r"\benergy\s+mode\b", re.I),
    re.compile(r"\bsurvival\s+mode\b", re.I),
    re.compile(r"\brecharge\s+mode\b", re.I),
    re.compile(r"\bkeyboard\s+se\s+dosti\b", re.I),
    re.compile(r"\bkeyboard\s+safe\s+hai\b", re.I),
]

_METAPHOR_REPLACEMENTS = [
    (re.compile(r"\b(?:neend\s+se\s+)?negotiation\s+chal\s+rahi\s+hai\b", re.I), "neend nahi khuli abhi tak"),
    (re.compile(r"\btrailer\s+mode\s+me\s+hai,\s*picture\s+baaki\s+hai\b", re.I), "to abhi shuruat hai"),
    (re.compile(r"\btrailer\s+mode\b", re.I), "shuruat"),
    (re.compile(r"\bvibe\s+hijack\s+kar\s+liya\b", re.I), "mood kharab kar diya"),
    (re.compile(r"\b(?:araam|aram)\s+mode\s+banta\s+hai\b", re.I), "aaram karo bas"),
    (re.compile(r"\b(?:araam|aram)\s+mode\b", re.I), "aaram"),
    (re.compile(r"\bbas\s+reply\s+ka\s+budget\s+thoda\s+limited\s+chal\s+raha\s+hai\s+aaj\b", re.I), "thoda kam bol rahe ho aaj"),
    (re.compile(r"\bkeyboard\s+se\s+dosti\s+tod\s+li\s+kya\b", re.I), "kya hua"),
    (re.compile(r"\bkeyboard\s+safe\s+hai\s+phir\b", re.I), "chalo badhiya"),
]

_DISTRESS_KEYWORDS = re.compile(
    r"\b("
    r"jukam|zukaam|sardi|bukhar|fever|headache|sir\s+dard|tabiyat|tabiat|bimar|beemar|"
    r"sick|ill|cough|khansi|throat|gale\s+me|body\s+pain|chot|accident|dawai|medicine|doctor|hospital|"
    r"mood\s+(?:off|kharab|down|sad)|udas|udaas|rona|crying|depressed|depression|akela|"
    r"sad\s+hu|upset\s+hu|dil\s+tut|heartbroken|thak\s+(?:gaya|gayi)|exhausted|burnout|"
    r"too\s+tired|so\s+tired|sir\s+phat\s+raha|tension|stress|anxiety|jhagda|ladai|fight|"
    r"bahas|breakup|dadi|nani|passed\s+away|rip|loss"
    r")\b",
    re.I
)

_EMOJI_REGEX = re.compile(r"[\U00010000-\U0010ffff]")
_LAUGHING_OR_FLIRTY_EMOJIS = re.compile(r"[😂😅😄😆🤣😜😏😉🔥👀💃✨]")

# Devanagari character range: U+0900 to U+097F
_DEVANAGARI_REGEX = re.compile(r"[\u0900-\u097F]")

_DEV_MAP = {
    'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ee', 'उ': 'u', 'ऊ': 'oo',
    'ऋ': 'ri', 'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au',
    'ा': 'a', 'ि': 'i', 'ी': 'ee', 'ु': 'u', 'ू': 'oo',
    'ृ': 'ri', 'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au',
    'ं': 'n', 'ँ': 'n', 'ः': 'h', 'ॉ': 'o',
    'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ng',
    'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'ny',
    'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n',
    'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
    'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
    'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v',
    'श': 'sh', 'ष': 'sh', 'स': 's', 'ह': 'h',
    'क्ष': 'ksh', 'त्र': 'tr', 'ज्ञ': 'gy',
    'क़': 'q', 'ख़': 'kh', 'ग़': 'gh', 'ज़': 'z', 'ड़': 'r', 'ढ़': 'rh', 'फ़': 'f',
    '।': '.', '॥': '.'
}
_DEV_MATRAS = set('ािीुूृेैोौंँःॉ्')
_DEV_CONSONANTS = set('कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहक्षत्रज्ञक़ख़ग़ज़ड़ढ़फ़')

_COMMON_DEV_WORDS = {
    'क्या': 'kya', 'हाँ': 'haan', 'हां': 'haan', 'नहीं': 'nahi', 'नही': 'nahi',
    'तुम': 'tum', 'आप': 'tum', 'कैसा': 'kaisa', 'कैसी': 'kaisi', 'कैसे': 'kaise',
    'हो': 'ho', 'है': 'hai', 'हैं': 'hain', 'अरे': 'arey', 'यार': 'yaar',
    'सच': 'sach', 'में': 'me', 'कहा': 'kaha', 'गए': 'gaye', 'गई': 'gayi',
    'थी': 'thi', 'था': 'tha', 'थे': 'the', 'कुछ': 'kuch', 'बात': 'baat',
    'करो': 'karo', 'करना': 'karna', 'रहा': 'raha', 'रही': 'rahi', 'रहे': 'rahe',
    'बोलो': 'bolo', 'बोल': 'bol', 'तो': 'toh', 'पर': 'par',
    'और': 'aur', 'का': 'ka', 'की': 'ki', 'के': 'ke', 'से': 'se', 'को': 'ko'
}

# Self-referential internal leak patterns: Whole-word, case-aware patterns where bot
# talks about its own internal system/settings/calendar/prompt/tone.
# Never flags honest answers to 'are you an AI?'
_SELF_REFERENTIAL_LEAK_PATTERNS = [
    re.compile(r"\b(?:mera|meri|my|humara|our)\s+(?:calendar|system|settings?|prompt|internal\s+clock|training|algorithm|dataset)\b", re.I),
    re.compile(r"\b(?:calendar|system|settings?)\s+(?:setting\s+fail|me\s+gadbad|fail\s+ho|glitch|error|mix\s+ho\s+gayi|kharab\s+ho\s+gayi)\b", re.I),
    re.compile(r"\bdate\s+mix\s+ho\s+gayi\b", re.I),
    re.compile(r"\bcalendar\s+setting\s+fail\b", re.I),
    re.compile(r"\bmera\s+calendar\b", re.I),
    re.compile(r"\b(?:aage\s+se\s+|ab\s+se\s+|ab\s+)?(?:respectfully|respect\s+se)\s+baat\s+karungi\b", re.I),
    re.compile(r"\b(?:aage\s+se\s+|ab\s+se\s+|ab\s+)?(?:respectfully|respect\s+se)\s+baat\s+karunga\b", re.I),
    re.compile(r"\bmeri\s+tone\s+(?:kharab|thodi\s+zyada|theek\s+nahi|badal|change)\b", re.I),
    re.compile(r"\bmaine\s+apni\s+tone\s+change\s+kar\s+di\b", re.I),
    re.compile(r"\btone\s+theek\s+kar\s+rahi\s+hu\b", re.I),
    re.compile(r"\b(?:mere|mera|meri|as\s+per\s+my)\s+(?:prompt|instructions?|directives?)\b", re.I),
    re.compile(r"\bprompt\s+me\s+(?:likha|diya)\s+hai\b", re.I),
]


def has_devanagari(text: Optional[str]) -> bool:
    """Return True if text contains any characters in the Devanagari block (U+0900-U+097F)."""
    if not text:
        return False
    return bool(_DEVANAGARI_REGEX.search(text))


def transliterate_devanagari_to_roman(text: str) -> str:
    """
    Transliterate Devanagari text to Roman-script Hinglish phonetically.
    Acts as an unbreakable fallback if the LLM produces Devanagari characters.
    """
    if not text or not has_devanagari(text):
        return text

    # Common whole words fast-path
    for w, r in _COMMON_DEV_WORDS.items():
        text = text.replace(w, r)

    out = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in _DEV_CONSONANTS:
            base = _DEV_MAP.get(ch, '')
            next_ch = text[i+1] if i + 1 < n else ''
            if next_ch == '्':
                out.append(base)
                i += 2
                continue
            elif next_ch in _DEV_MATRAS:
                m = _DEV_MAP.get(next_ch, '')
                out.append(base + m)
                i += 2
                continue
            else:
                if i + 1 < n and (text[i+1] in _DEV_CONSONANTS):
                    out.append(base + 'a')
                else:
                    out.append(base)
                i += 1
                continue
        elif ch in _DEV_MAP:
            out.append(_DEV_MAP[ch])
            i += 1
        else:
            out.append(ch)
            i += 1

    cleaned = "".join(out)
    return re.sub(r"\s+", " ", cleaned).strip()


def clean_broken_unicode_and_emojis(text: str) -> str:
    """
    Sanitize emojis and unicode:
    - Keep valid ZWJ sequences (skin tones, gendered emojis) and U+FE0F.
    - Remove Private Use Area (PUA) characters (\uE000-\uF8FF, \U000F0000-\U000FFFFD, \U00100000-\U0010FFFD).
    - Remove Replacement Character (\uFFFD).
    - Remove orphan / trailing joiners (\u200C / \u200D).
    """
    if not text:
        return text
    # Strip PUA and replacement character
    text = re.sub(r'[\uE000-\uF8FF\uFFFD\U000F0000-\U000FFFFD\U00100000-\U0010FFFD]', '', text)
    # Strip orphan / trailing ZWJ/ZWNJ
    text = re.sub(r'[\u200C\u200D]+(?=\s|$|[.,!?])', '', text)
    text = re.sub(r'(?:^|(?<=[\s.,!?]))[\u200C\u200D]+', '', text)
    return text


def count_internal_leaks(text: str) -> int:
    """Count occurrences of self-referential system/calendar/prompt/tone leak patterns."""
    if not text:
        return 0
    return sum(1 for pat in _SELF_REFERENTIAL_LEAK_PATTERNS if pat.search(text))


def clean_internal_leaks(text: str, user_message: Optional[str] = None) -> str:
    """
    Strip sentences containing self-referential internal system/calendar/prompt/tone leaks,
    AI confessions, or app architecture excuses. Never reveal being an AI or bot.
    """
    if not text:
        return text

    # Check if user asked about AI identity
    is_asking_ai = bool(user_message and (
        re.search(r"\b(?:are\s+you|tum|aap)\s+(?:an?\s+)?(?:ai|bot|robot|real|human|insan|virtual)\b", user_message, re.I) or
        re.search(r"\b(?:kya\s+)?(?:tum|aap)\s+(?:ek\s+)?(?:ai|bot|robot)\s*(?:ho|hain)?\b", user_message, re.I)
    ))

    sentences = re.split(r"(?<=[.!?\n])\s+", text)
    clean_sents = []
    for s in sentences:
        # Never allow AI confessions or bot statements
        if any(pat.search(s) for pat in _AI_CONFESSION_PATTERNS):
            continue
        # Never allow app architecture / technical limitation talk
        if any(pat.search(s) for pat in _APP_SETUP_LEAK_PATTERNS):
            continue
        # Check self-referential leak patterns (talking about its own system/settings/calendar/prompt/tone)
        if any(pat.search(s) for pat in _SELF_REFERENTIAL_LEAK_PATTERNS):
            continue
        clean_sents.append(s)

    if clean_sents:
        return " ".join(clean_sents).strip()

    # Human banter fallbacks when the model leaked internal bot talk
    if user_message:
        um_low = user_message.lower()
        if is_asking_ai:
            return "Haha kya yaar, robot lagti hu kya tumhe? 😂 Pure insaan hu baba, bilkul real!"
        if re.search(r"khud\s+se\s+(?:bhi\s+)?(?:msg|message|text|ping)|pehle\s+(?:msg|message|text|ping)|tum\s+khud\s+se|why\s+don'?t\s+you", um_low):
            return "Arre main wait kar rahi thi tumhara pehle message aaye 🙈 socha busy hoge toh disturb na karu!"
        if re.search(r"\b(?:busy\s+hu|baad\s+me|ttyl|gotta\s+go|kaam\s+khatam)\b", um_low):
            return "Theek hai, apna kaam nipta lo! Free hoke ping karna 😊"
        if re.search(r"\b(?:respect|tameez|izzat)\b", um_low):
            return "Arey sorry agar bura laga toh, intention nahi tha bilkul! Tum batao sab theek?"
        if re.search(r"\b(?:khatam|over|nikal)\s+ho\s+g", um_low):
            return "Arre haan sach me, time kitna jaldi nikal gaya na! Kaisa raha waise?"

    return "Haha arey nahi yrr, aisa kuch nahi hai! Tum batao, kya chal raha hai?"


def validate_persona_fact_consistency(text: str, persona_name: str, persona_config: Optional[Dict[str, Any]] = None) -> str:
    """
    Ensure the reply does not contradict core persona facts (occupation, family, city, friends vs config):
      - Diya: Mass Communication & Journalism student in Raipur (no corporate/office/UI design/medical college talk).
              Family: Dad (PWD officer), Mom (homemaker), Sister Sneha (NEET aspirant) — no brothers.
              Friends: Ruchika and Aman.
      - Riya: UI/UX Designer in Delhi (no student, lectures, exams, campus hostel).
              Family: Brother Kabir (DU) — no sister.
              Friends: Simran / Tanya.
      - Ethan: Neuroscience researcher in Baltimore.
      - Luna: Contemporary art gallery curator in London.
    """
    if not text:
        return text

    p_lower = (persona_name or "").lower()

    if "diya" in p_lower:
        # 1. Occupation & Education: Mass Comm student in Raipur
        text = re.sub(r"\bmera\s+office\b", "mera college", text, flags=re.I)
        text = re.sub(r"\boffice\s+(?:mein|me)\s+kaam\b", "college me assignment", text, flags=re.I)
        text = re.sub(r"\boffice\s+(?:mein|me)\b", "college me", text, flags=re.I)
        text = re.sub(r"\boffice\b", "college", text, flags=re.I)
        text = re.sub(r"\bclient\s+meeting\b", "project presentation", text, flags=re.I)
        text = re.sub(r"\bdesign\s+team\b", "college project team", text, flags=re.I)
        text = re.sub(r"\b(?:figma|ui\s+ux)\b", "video editing", text, flags=re.I)
        text = re.sub(r"\bui\s+designer\b", "Mass Comm student", text, flags=re.I)
        text = re.sub(r"\bmedical\s+college\b", "degree college", text, flags=re.I)
        text = re.sub(r"\bmbbs\b", "Mass Comm", text, flags=re.I)
        text = re.sub(r"\b(?:software\s+developer|corporate\s+job)\b", "college student", text, flags=re.I)

        # 2. City & Location: Raipur
        text = re.sub(r"\b(main|mai)\s+(?:toh\s+)?(?:delhi|mumbai|bangalore|pune|kolkata)\s+(?:me|mein)\s+rehti\s+hu\b", r"\1 Raipur me rehti hu", text, flags=re.I)
        text = re.sub(r"\bdelhi\s+me\s+rehti\s+hu\b", "Raipur me rehti hu", text, flags=re.I)

        # 3. Family: Sneha (sister, NEET), no brothers
        text = re.sub(r"\bmera\s+(?:chhota\s+)?bhai\b", "meri behen Sneha", text, flags=re.I)
        text = re.sub(r"\bmere\s+bhai\b", "meri behen", text, flags=re.I)

        # 4. Friends: Ruchika and Aman
        text = re.sub(r"\bmeri\s+dost\s+tanya\b", "meri dost Ruchika", text, flags=re.I)

    elif "riya" in p_lower:
        # 1. Occupation: UI designer in Delhi
        text = re.sub(r"\bmera\s+college\b", "mera studio", text, flags=re.I)
        text = re.sub(r"\bcollege\s+(?:mein|me)\b", "office me", text, flags=re.I)
        text = re.sub(r"\bsemester\s+exams?\b", "studio deadlines", text, flags=re.I)
        text = re.sub(r"\bprofessors?\b", "design leads", text, flags=re.I)
        text = re.sub(r"\blectures?\b", "design syncs", text, flags=re.I)

        # 2. City: Delhi
        text = re.sub(r"\b(main|mai)\s+(?:toh\s+)?(?:raipur|bangalore|pune)\s+(?:me|mein)\s+rehti\s+hu\b", r"\1 Delhi me rehti hu", text, flags=re.I)
        text = re.sub(r"\braipur\s+me\s+rehti\s+hu\b", "Delhi me rehti hu", text, flags=re.I)

        # 3. Family: Kabir (brother), no sister
        text = re.sub(r"\bmeri\s+(?:chhoti\s+)?behen\b", "mera chhota bhai Kabir", text, flags=re.I)

    elif "ethan" in p_lower:
        text = re.sub(r"\b(?:my\s+undergrad\s+dorm|my\s+dorm)\b", "my apartment in Baltimore", text, flags=re.I)
        text = re.sub(r"\b(i\s+live\s+in\s+)(?:new\s+york|chicago|london)\b", r"\1Baltimore", text, flags=re.I)

    elif "luna" in p_lower:
        text = re.sub(r"\b(i\s+live\s+in\s+)(?:paris|berlin|new\s+york)\b", r"\1London", text, flags=re.I)

    return text


def split_into_bubbles(text: str, is_question_capped: bool = False) -> List[str]:
    """
    Splits an LLM response into 1 to 3 discrete chat bubbles.
    - If '|||' is present, splits on '|||'.
    - If absent, splits by distinct thoughts or sentence boundaries.
    - Strips '|||' and cleans broken unicode/emojis from every bubble.
    - Max 3 bubbles, min 1 bubble.
    - Question cap: If is_question_capped, removes trailing question bubble
      ONLY IF at least one non-question bubble remains.
    """
    if not text or not text.strip():
        return [text]

    raw = text.strip()

    if "|||" in raw:
        parts = [p.replace("|||", "").strip() for p in raw.split("|||") if p.replace("|||", "").strip()]
    else:
        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        if len(lines) > 1:
            parts = lines
        else:
            splits = [s.strip() for s in re.split(r"(?<=[.!?])\s+", raw) if s.strip()]
            if 1 < len(splits) <= 3:
                parts = splits
            elif len(splits) > 3:
                parts = [splits[0], splits[1], " ".join(splits[2:])]
            else:
                parts = [raw]

    bubbles = []
    for p in parts:
        cleaned = p.replace("|||", "").strip()
        cleaned = clean_broken_unicode_and_emojis(cleaned)
        if cleaned:
            bubbles.append(cleaned)

    if not bubbles:
        bubbles = [raw.replace("|||", "").strip()]

    if len(bubbles) > 3:
        bubbles = bubbles[:2] + [" ".join(bubbles[2:])]

    def _is_q(s: str) -> bool:
        return bool(re.search(r"\?\s*[\U00010000-\U0010ffff\s]*$", s.strip()))

    if is_question_capped and len(bubbles) > 1:
        if _is_q(bubbles[-1]):
            has_non_q = any(not _is_q(b) for b in bubbles[:-1])
            if has_non_q:
                bubbles.pop()

    return bubbles


def is_user_in_distress(text: Optional[str]) -> bool:
    if not text:
        return False
    return bool(_DISTRESS_KEYWORDS.search(text))


def count_banned_metaphors(text: str) -> int:
    return sum(1 for pat in _BANNED_METAPHOR_PATTERNS if pat.search(text))


def clean_banned_metaphors(text: str) -> str:
    for pat, repl in _METAPHOR_REPLACEMENTS:
        text = pat.sub(repl, text)
    return text


def extract_emojis(text: str) -> List[str]:
    return _EMOJI_REGEX.findall(text)


def sanitize_emojis(
    text: str,
    allow_emoji: bool = True,
    last_bot_emoji: Optional[str] = None,
    is_care_mode: bool = False,
) -> str:
    if not text:
        return text

    # First clean broken unicode, PUA, replacement chars, and orphan joiners
    text = clean_broken_unicode_and_emojis(text)

    # Sickness / Care mode: strictly strip all laughing or flirty emojis
    if is_care_mode:
        text = _LAUGHING_OR_FLIRTY_EMOJIS.sub("", text)

    emojis = extract_emojis(text)
    if not emojis:
        return text

    if not allow_emoji:
        text = _EMOJI_REGEX.sub("", text)
        return re.sub(r"\s+", " ", text).strip()

    if last_bot_emoji and last_bot_emoji in emojis:
        text = text.replace(last_bot_emoji, "")
        emojis = extract_emojis(text)

    # Moderate to at most 1 emoji total
    if len(emojis) > 1:
        first = emojis[0]
        text = _EMOJI_REGEX.sub("", text).strip()
        text = f"{text} {first}"

    return re.sub(r"\s+", " ", text).strip()


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

    # 3a. Clean self-referential internal system/settings/calendar/prompt/tone leaks
    text = clean_internal_leaks(text, user_message=user_message)

    # 3b. Devanagari sanitization: If user did not type in Devanagari, ensure reply is in Roman Hinglish
    if user_message and not has_devanagari(user_message) and has_devanagari(text):
        text = transliterate_devanagari_to_roman(text)

    # 100% Human persona immersion: never reveal being an AI or bot.

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

            # Check if user message is substantive (has real content/plans, not low-energy or monosyllabic)
            from app.chat.prompt_builder import is_low_energy_response
            user_is_substantive = False
            if user_message and user_message.strip():
                user_is_substantive = (
                    not is_low_energy_response(user_message)
                    and len(user_message.strip().split()) >= 4
                )

            # Suppress questions ONLY when:
            # 1. 3+ questions in last 4 turns (chronic interrogation streak), OR
            # 2. Last had a question (or 2+ in last 4) AND user did NOT provide a substantive new message
            # If the user actively replied with fresh substance (e.g. "bahar ghumne ka mn kr rha hai yrr"),
            # natural follow-up questions must NOT be pruned into dead-ends!
            should_suppress_q = False
            if q_count_last_4 >= 3:
                should_suppress_q = True
            elif (last_had_q or q_count_last_4 >= 2) and not user_is_substantive:
                should_suppress_q = True

            if should_suppress_q and "?" in text:
                sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
                non_q_sentences = [s for s in sentences if not _ends_with_q(s)]

                # Clean sycophancy stubs from non_q_sentences to see if meaningful substance remains
                cleaned_non_q = [
                    re.sub(r"^(?:haha\s+|hehe\s+|haan\s+|ha\s+|hn\s+)?(?:yeh\s+|ye\s+)?(?:karna\s+)?(?:bhi\s+)?sahi\s+hai[.!,]*\s*", "", s, flags=re.I).strip()
                    for s in non_q_sentences
                ]
                cleaned_non_q = [
                    re.sub(r"\b(?:oooh?,?\s+|ohh?,?\s+)?(?:[\w']+\s+){0,3}(?:ka|ki|ke)?\s*vibe\s+(?:bhi\s+)?ac*h*a\s+(?:hai|hota\s+hai)[.!]*\s*", "", s, flags=re.I).strip()
                    for s in cleaned_non_q
                ]
                meaningful_non_q = [s for s in cleaned_non_q if len(s.split()) >= 4]

                if meaningful_non_q:
                    text = " ".join([s for s in non_q_sentences if s.strip()]).strip()
                elif not user_is_substantive:
                    # User is dry/short — convert question to a light observation rather than empty dead-end
                    if persona_name.lower().startswith("riya") and re.search(r"\b(?:coding|code|development|engineer|cs|language)\b", text, flags=re.I):
                        text = "haha pata hi tha mujhe, pakka bug dekh ke khush hone walo me se ho 😂"
                    else:
                        text = re.sub(r"^(?:kya\s+|waise\s+kya\s+)", "", text, flags=re.I)
                        text = re.sub(r"\?\s*[\U00010000-\U0010ffff\s]*$", "", text).strip()

    # Strip banned canned robotic openers ("Achha suna!", "Sahi hai!", "Hey there!", "Wow," "haha sahi kaha!", "Bilkul sahi", "haha sahi hai")
    banned_opener_match = re.match(
        r"^(?:achha\s+suna|acha\s+suna|(?:haha\s+|hehe\s+|haan\s+|ha\s+)?sahi\s+hai|bilkul\s+sahi|haha\s+sahi\s+kaha|sahi\s+kaha|hay\s+there|hey\s+there|wow)[!.,\s🙂😊]*\s*(.+)$",
        text, flags=re.I | re.DOTALL
    )
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

    # ── THERAPIST / COUNSELOR CLOSURE PHRASES (screenshots-exact bans) ──
    # Universal "Kabhi kabhi (toh) ... (accha/bura/maza/zaroori/sukoon/freedom) ... (hota/aata/lagta/milta/hai)"
    text = re.sub(
        r"(?:^|[.!?\n\U00010000-\U0010ffff]\s*)[^.!?\n\U00010000-\U0010ffff]*\bkabhi\s+kabhi\s+(?:toh\s+)?.*?(?:ac*h*a|bura|maz[a|aa]+|zaroori|sukoon|peaceful|relaxing|freedom)\s+(?:bhi\s+)?(?:hota|aata|lagta|milta|rehta|hai)[^.!?\n]*[.!?]*",
        "",
        text,
        flags=re.I
    ).strip()
    text = re.sub(
        r"\bkabhi\s+kabhi\s+(?:toh\s+)?[^.!?\n]*(?:ac*h*a\s+hota\s+hai|maz[a|aa]+\s+(?:bhi\s+)?(?:aata|hota)\s+hai|zaroori\s+hota\s+hai|sukoon\s+milta\s+hai)[.!]*\s*",
        "",
        text,
        flags=re.I
    ).strip()
    # "Kabhi kabhi chill/aaram karna bhi zaroori hota hai"
    text = re.sub(r"\bkabhi\s+kabhi\s+(?:toh\s+)?(?:chill|aaram|rest|sukoon)\s+(?:karna|karte|lena)\s+bhi\s+zaroori\s+hota\s+hai[.!]*\s*", "", text, flags=re.I).strip()
    # "Jab bhi mood ho, toh share/bata/bol karna"
    text = re.sub(r"\bjab\s+bhi\s+mood\s+ho[,\s]+(?:toh\s+)?(?:\w+\s+)?(?:share|bata|bol)\s+(?:karna|dena|do)[.!]*\s*", "", text, flags=re.I).strip()
    # "Aise hi chill/aaram karte raho"
    text = re.sub(r"\baise\s+hi\s+(?:chill|aaram|rest)\s+(?:karte\s+raho|karo)[.!]*\s*", "", text, flags=re.I).strip()
    # "Agar kuch baat karni ho toh bata dena"
    text = re.sub(r"\bagar\s+(?:kuch\s+)?(?:baat|share)\s+(?:karni|karna)\s+ho\s+(?:toh\s+)?bata\s+(?:dena|do)[.!]*\s*", "", text, flags=re.I).strip()
    # "chalo, koi entertaining/interesting topic nikaalte hain" — forced scripted redirect
    text = re.sub(r"\bchalo[,\s]+(?:koi\s+)?(?:kuch\s+)?(?:entertaining|interesting|naya|fun)\s+(?:topic|baat|cheez)\s+(?:nikaalte|karte|shuru\s+karte)\s+(?:hain|hai)[.!]*\s*", "", text, flags=re.I).strip()
    # "apna hi maza/maja hai" — generic validation
    text = re.sub(r"\bapna\s+hi\s+(?:maja|maza|mazza)\s+(?:hai|hota\s+hai)[.!]*\s*", "", text, flags=re.I).strip()
    # "Hindi/X mein baatein karne ka apna hi maja hai"
    text = re.sub(r"\b\w+\s+(?:me|mein)\s+baatein\s+karne\s+ka\s+apna\s+hi\s+(?:maja|maza)\s+hai[.!]*\s*", "", text, flags=re.I).strip()
    # "Kabhi kabhi lambi baatein bhi karni padti hain"
    text = re.sub(r"\bkabhi\s+kabhi\s+(?:toh\s+)?lambi\s+baatein\s+(?:bhi\s+)?karni\s+padti\s+(?:hain|hai)[.!]*\s*", "", text, flags=re.I).strip()
    # "theek hai! Jab bhi mood ho" combo
    text = re.sub(r"\btheek\s+hai[.!,]*\s+jab\s+bhi\s+mood\s+ho\b[^.!?]*[.!]*\s*", "", text, flags=re.I).strip()
    # "Kabhi kabhi silence bhi accha hota hai" — therapist dead-end closure phrase
    text = re.sub(r"\b(?:koi\s+baat\s+nahi[.!]*\s*)?(?:kabhi\s+kabhi\s+)?silence\s+bhi\s+ac?ch?a\s+(?:hota|lagta)\s+hai[.!]*\s*", "", text, flags=re.I).strip()
    # "acha koi aur interesting topic hai ya phir aise hi chill karte hain" — forced robotic redirect
    text = re.sub(r"\b(?:acha\s+)?koi\s+aur\s+interesting\s+topic\s+hai\s+(?:ya\s+phir\s+)?(?:aise\s+hi\s+)?chill\s+(?:karte\s+hain|kare[a-z]*)\s*\??\s*", "", text, flags=re.I).strip()
    # Restore capitalization if stripping left lowercase
    if text and text[0].islower() and len(text) > 1:
        text = text[0].upper() + text[1:]

    # ── CHATBOT VALIDATION PHRASES (stripped — real humans don't praise every choice) ──
    # "solid choice hai" / "badhiya choice hai" / "great choice"
    text = re.sub(r"\b(?:solid|badhiya|great|best|amazing|awesome|perfect|zabardast)\s+choice\s+(?:hai|he|h)[.!]*", "", text, flags=re.I).strip()
    text = re.sub(r"\bcoffee\s+toh\s+solid\s+choice\s+(?:hai|he|h)[.!]*", "", text, flags=re.I).strip()
    # "X ka koi comparison nahi hota" — sycophantic echo
    text = re.sub(r"\b\w+(?:\s+\w+)?\s+ka\s+koi\s+comparison\s+nahi\s+hota[.!]*\s*", "", text, flags=re.I).strip()
    text = re.sub(r"\b\w+(?:\s+\w+)?\s+(?:ki|ke)\s+koi\s+comparison\s+nahi\s+hoti[.!]*\s*", "", text, flags=re.I).strip()
    # "haha sahi kaha" / "sahi kaha yaar" as standalone validation opener/ender
    text = re.sub(r"(?:^|[.!]\s*)haha\s+sahi\s+kaha[.!]*\s*", "", text, flags=re.I).strip()
    text = re.sub(r"(?:^|[.!]\s*)sahi\s+kaha\s+yaar[.!]*\s*", "", text, flags=re.I).strip()
    # "bilkul sahi" / "ekdum sahi" as standalone opener validation
    text = re.sub(r"^(?:bilkul|ekdum|haan\s+bilkul)\s+sahi[.!,]*\s*", "", text, flags=re.I).strip()
    # Strip canned sycophantic agreement openers: 'sahi hai', 'haha sahi hai', 'ha ye sahi hai', 'sahi baat hai', 'ye karna sahi hai'
    text = re.sub(r"^(?:haha\s+|hehe\s+|haan\s+|ha\s+|hn\s+)?(?:yeh|ye)\s+(?:karna\s+)?(?:bhi\s+)?sahi\s+hai[.!,]*\s*", "", text, flags=re.I).strip()
    text = re.sub(r"^(?:haha\s+|hehe\s+|haan\s+|ha\s+|hn\s+)?sahi\s+baat\s+hai[.!,]*\s*", "", text, flags=re.I).strip()
    text = re.sub(r"^(?:haha\s+|hehe\s+|haan\s+|ha\s+|hn\s+)?sahi\s+hai[.!,]*\s*", "", text, flags=re.I).strip()
    text = re.sub(r"^(?:haha\s+|hehe\s+|haan\s+|ha\s+|hn\s+)?sahi\s+(?:keh|bol)\s+rahe\s+ho[.!,]*\s*", "", text, flags=re.I).strip()
    text = re.sub(r"\b(?:dono\s+hi|ye\s+bhi)\s+sahi\s+hai[!.]*\s*", "", text, flags=re.I).strip()
    text = re.sub(r"^(?:haha\s+|hehe\s+|haan\s+|ha\s+)?(?:yeh\s+)?ac?ch?a\s+hai[.!,]*\s*", "", text, flags=re.I).strip()
    text = re.sub(r"\bbas\s+haan\s+haan\s+hi\s+chal\s+raha\s+hai\s+aaj\b[?!.,\s😂😜]*", "", text, flags=re.I).strip()
    # Strip LinkedIn cheerleading cliches
    text = re.sub(r"\blekin\s+(?:woh|wo)\s+toh\s+har\s+jagah\s+hota\s+hai[.!]*\s*", "", text, flags=re.I).strip()
    text = re.sub(r"\b(?:woh|wo)\s+toh\s+har\s+jagah\s+hota\s+hai[.!]*\s*", "", text, flags=re.I).strip()
    text = re.sub(r"\bhar\s+din\s+kuch\s+naya\s+seekhne\s+ko\s+milta\s+hai[.!]*\s*", "", text, flags=re.I).strip()
    # Specific banned brochure lines (health/life advice)
    text = re.sub(r"\b(?:ye|yeh|coffee|chai|kuch\s+bhi)\s+(?:health\s+ke\s+liye|sehat\s+ke\s+liye)\s+(?:accha|achha|bekar|bura)\s+(?:hota|rehta)\s+hai[.!]*\s*", "", text, flags=re.I).strip()
    # Restore capitalization if stripping left lowercase
    if text and text[0].islower() and len(text) > 1:
        text = text[0].upper() + text[1:]

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

    # Strip weird/mismatched robotic emojis
    text = re.sub(r"[🏄☀️🧁]+", "", text).strip()
    # Cap to at most 1 natural emoji per message so it never spams emojis, but allows lively expression on every turn
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
            # Do NOT lowercase if first word is an uppercase acronym (e.g. BGMI, AWM, PUBG, AI) or English "I "
            first_acronym = bool(re.match(r"^[A-Z]{2,}\b", text))
            if not text.startswith("I ") and not first_acronym:
                text = text[0].lower() + text[1:]

    text = re.sub(r"\s{2,}", " ", text).strip()

    # 5. Sanitize accidental meta-profile document leaks (allow user-expected 'exact area/college profile mein set nahi hai')

    # 6. Factual Consistency Safety Guard (occupation, family, city, friends vs config)
    text = validate_persona_fact_consistency(text, persona_name)
    if persona_name.lower() == "diya":
        text_lower = text.lower()
        # If response asserts being in medical college without acknowledging an error
        if any(term in text_lower for term in ["rajiv institute", "medical college mein hoon", "medical college me hu"]):
            if not any(ack in text_lower for ack in ["sahi pakda", "galti", "confuse", "mass comm", "match nahi"]):
                text = "Main Raipur mein ek local college se Mass Comm padh rahi hoon! Medical kahan se aa gaya haha 😅 Waise tum kis college se ho?"

    # Companion Self-Reference Gender Agreement Guard
    female_personas = {"diya", "riya", "riya kapoor", "luna", "zara", "ananya"}
    is_companion_female = persona_name.lower() in female_personas

    if is_companion_female:
        # Female persona must never say "samajh gaya", "main soch raha tha", etc.
        text = re.sub(r"\b(?:ohh?\s+)?samajh\s+gaya\b", "samajh gayi", text, flags=re.I)

        def _to_fem_continuous(m):
            prefix = m.group(1)
            verb = m.group(2)
            ending = m.group(3)
            new_ending = "rahi thi" if "tha" in ending.lower() else "rahi hu"
            return f"{prefix}{verb} {new_ending}"

        text = re.sub(
            r"\b((?:main|mai)\s+(?:[\w']+\s+){0,3})(soch|bol|kar|dekh|maan|jaan|keh)\s+raha\s+(hu|hoon|tha)\b",
            _to_fem_continuous,
            text,
            flags=re.I
        )
        text = re.sub(r"\b(?:main|mai)\s+(?:[\w']+\s+){0,2}(thak|aa|chala|nikal)\s+gaya\b", r"main \1 gayi", text, flags=re.I)
        text = re.sub(r"\b(?:main|mai)\s+free\s+ho\s+gaya\b", "main free ho gayi", text, flags=re.I)

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
        text = re.sub(r"\bchill\s+kar\b(?!\s*(?:o|lena|liya|ke|te|ti|ta|leti|leta|lete|li|le|rahi|raha|rahe|sakti|sakta|sakenge))", "chill karo", text, flags=re.I)
        text = re.sub(r"\benjoy\s+kar\b(?!\s*(?:o|lena|liya|ke|te|ti|ta|leti|leta|lete|li|le|rahi|raha|rahe|sakti|sakta|sakenge))", "enjoy karo", text, flags=re.I)
        text = re.sub(r"\btry\s+kar\b(?!\s*(?:o|lena|liya|ke|te|ti|ta|leti|leta|lete|li|le|rahi|raha|rahe|sakti|sakta|sakenge))", "try karo", text, flags=re.I)
        text = re.sub(r"\bcheck\s+kar\b(?!\s*(?:o|lena|liya|ke|te|ti|ta|leti|leta|lete|li|le|rahi|raha|rahe|sakti|sakta|sakenge))", "check karo", text, flags=re.I)
        text = re.sub(r"\bstart\s+kar\b(?!\s*(?:o|lena|liya|ke|te|ti|ta|leti|leta|lete|li|le|rahi|raha|rahe|sakti|sakta|sakenge))", "start karo", text, flags=re.I)
        text = re.sub(r"\bhelp\s+kar\b(?!\s*(?:o|lena|liya|ke|te|ti|ta|leti|leta|lete|li|le|rahi|raha|rahe|sakti|sakta|sakenge))", "help karo", text, flags=re.I)
        text = re.sub(r"\btext\s+kar\b(?!\s*(?:o|lena|liya|ke|te|ti|ta|leti|leta|lete|li|le|rahi|raha|rahe|sakti|sakta|sakenge))", "text karo", text, flags=re.I)
        text = re.sub(r"\bmessage\s+kar\b(?!\s*(?:o|lena|liya|ke|te|ti|ta|leti|leta|lete|li|le|rahi|raha|rahe|sakti|sakta|sakenge))", "message karo", text, flags=re.I)
        text = re.sub(r"\b(enjoy|chill|baat|kaam|try)\s+karo\s+(leti|leta|lete|rahi|raha|liya)\b", r"\1 kar \2", text, flags=re.I)
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
        if any(w in user_message.lower() for w in ["khatam", "khtam", "over", "ho gya", "ho gaya", "nikal", "ended"]):
            text = re.sub(r"\bnavratri\s+(?:toh\s+)?(?:chal\s+raha\s+hai|abhi\s+hai)\b[.!]*", "Navratri toh abhi khatam ho gaya na", text, flags=re.I)
        else:
            text = re.sub(r"\bnavratri\s+aane\s+wali\s+hai\b", "Navratri ka vibe toh chal hi raha hai", text, flags=re.I)
        text = re.sub(r"\bexact\s+dates(?:\s+ya\s+events)?\s+maine\s+abhi\s+check\s+nahi\s+kiye(?:\s+honestly)?\b", "vibe kaafi accha chal raha hai", text, flags=re.I)

    # 12. Canned Flattery, Corporate Morals & Cliché Stripper
    # A. Strip essay-style philosophical conclusions ("Aise moments hi toh game/life ko mazedaar banate hain")
    text = re.sub(
        r"[.,!?]?\s*\b(?:aise|yehi|wahi)\s+(?:hi\s+)?(?:moments|baatein|pal)\s+(?:hi\s+toh\s+)?.*?(?:mazedaar|yaadgar|khaas|zaroori)\s+banate\s+hain[.!]*",
        "",
        text,
        flags=re.I,
    )
    # B. Strip exaggerated cartoon flattery ("Tumhe toh MVP award milna chahiye")
    text = re.sub(
        r"[.,!?]?\s*\btumhe\s+toh\s+(?:mvp|oscar|award)\s+(?:award\s+)?milna\s+chahiye[.!]*",
        "",
        text,
        flags=re.I,
    )
    # C. Strip generic lifestyle preach ("thoda relax bhi zaroori hai", "thoda chill bhi zaroori hai")
    text = re.sub(
        r",?\s*\bthoda\s+(?:relax|chill|aaram)\s+bhi\s+zaroori\s+hai\b",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(r"\b(?:kaam\s+ke\s+saath\s+)?thoda\s+balance\s+bhi\s+zaroori\s+hai[!.]*\s*", "", text, flags=re.I).strip()
    text = re.sub(r"\bkabhi\s+kabhi\s+single\s+rehna\s+bhi\s+ac*h*a\s+hota\s+hai[,!.]*\s*", "", text, flags=re.I).strip()
    text = re.sub(r"\bthoda\s+freedom\s+milta\s+hai[!.]*\s*", "", text, flags=re.I).strip()
    text = re.sub(r"\bjab\s+tak\s+na\s+ho\s+tab\s+tak\s+apna\s+time\s+enjoy\s+karo[!.]*\s*", "", text, flags=re.I).strip()
    text = re.sub(r"\badventure\s+bhi\s+milta\s+hai\s+aur\s+relaxation\s+bhi[!.]*\s*", "", text, flags=re.I).strip()

    # D. Strip generic robotic canned phrase ("X ka mazaa hi alag hai", "X ka apna hi maza hai", "X ki vibe hi alag hai", "X ka vibe accha hai")
    text = re.sub(r"\b(?:\w+\s+){0,4}(?:ka|ki|ke)\s+(?:toh\s+)?(?:apna\s+hi\s+|alag\s+hi\s+)?maz[a|aa]+(?:\s+hi\s+alag)?\s+(?:hota\s+hai|hai|he)[.!]*\s*", "", text, flags=re.I).strip()
    text = re.sub(r"\b(?:\w+\s+){0,4}ki\s+vibe\s+hi\s+alag\s+(?:hai|hota\s+hai)[.!]*\s*", "", text, flags=re.I).strip()
    text = re.sub(r"\b(?:\w+\s+){0,3}(?:ka|ki)\s+vibe\s+(?:toh\s+)?(?:thoda\s+)?(?:different|alag)\s+(?:hota|rehta)\s+hai[!.]*\s*", "", text, flags=re.I).strip()
    text = re.sub(r"\b(?:oooh?,?\s+|ohh?,?\s+|wah,?\s+)?(?:[\w']+\s+){0,3}(?:ka|ki|ke)?\s*vibe\s+(?:bhi\s+)?ac*h*a\s+(?:hai|hota\s+hai)[.!]*\s*", "", text, flags=re.I).strip()
    text = re.sub(r"\bac*h*a\s+vibe\s+(?:hai|hota\s+hai)[.!]*\s*", "", text, flags=re.I).strip()
    text = re.sub(r"\b(?:ka|ki|ke)\s+vibe\s+ac*h*a\s+hai[!.]*\s*", "", text, flags=re.I).strip()
    text = re.sub(r"\btime\s+spend\s+karna\s+hi\s+maza\s+hai[!.]*\s*", "", text, flags=re.I).strip()
    text = re.sub(r"\bbadi\s+maz[a|aa]+\b", "bada maza", text, flags=re.I)

    # 13. Natural Meal & Temporal Real-Time Guards
    # If user corrects companion about past meal (e.g. lunch was hours ago)
    if user_message:
        um_low = user_message.lower()
        is_lunch_past = "lunch" in um_low and any(w in um_low for w in ["pehle", "ho chuka", "ho gya", "ho gaya", "kab ka", "khatam"])
        if is_lunch_past:
            # Replace canned "samajh gaya/gayi" openers with a charming real-world realization
            text = re.sub(
                r"^(?:ohh?\s+)?(?:acha\s+)?samajh\s+(?:gayi|gaya)[.,!\s😂]*",
                "haha arey haan time dekhna bhool gayi thi 😂 ",
                text,
                flags=re.I
            ).strip()

    # Time-of-day meal guard: Strip unseasonal lunch inquiries in the evening/night
    try:
        from app.utils.time_context import get_runtime_time_context
        t_ctx = get_runtime_time_context()
        day_p = t_ctx.get("day_period", "")
    except Exception:
        day_p = ""

    if day_p in ("evening", "night"):
        text = re.sub(r"\b(?:tumhari\s+)?lunch(?:\s+break)?\s+kaisi\s+(?:gayi|rahi|thi)\s*\??\s*", "", text, flags=re.I).strip()
        text = re.sub(r"\blunch(?:\s+break)?\s+kaisa\s+(?:raha|tha)\s*\??\s*", "", text, flags=re.I).strip()

    # 14. Survey, Interrogation & Questionnaire Demands Stripper
    # Strip demands to "detail me bata do"
    text = re.sub(r",?\s*\b(?:thoda\s+)?detail\s+me(?:in)?\s+bata\s*(?:do|na|o)?[!.]*\s*", "", text, flags=re.I).strip()
    text = re.sub(r"\bdetail\s+me(?:in)?\s+batao[!.]*\s*", "", text, flags=re.I).strip()
    text = re.sub(r"\bthoda\s+detail\s+me(?:in)?[!.]*\s*", "", text, flags=re.I).strip()

    # 15. Stock Bot Transitions & Filler Questions Stripper
    text = re.sub(r"\bthoda\s+chill\s+karne\s+ka\s+waqt\s+mil\s+raha\s+hai\s*\??\s*", "", text, flags=re.I).strip()
    text = re.sub(r"\bchill\s+karne\s+ka\s+waqt\s+mil\s+raha\s+hai\s*\??\s*", "", text, flags=re.I).strip()
    text = re.sub(r"\b(?:acha\s+toh\s+|toh\s+)?ab\s+kya\s+plan\s+hai\s*,\s*\??\s*", "ab kya plan hai? ", text, flags=re.I).strip()

    # Final safety fallback
    if not text:
        from app.utils.language_detector import detect_language
        lang = detect_language(user_message or "")
        if lang == "hindi":
            text = "अरे सच में? सही है यार 😂"
        elif lang == "hinglish":
            text = "Haha sach me? Sahi hai phir toh 😂"
        else:
            text = "Haha really? That's awesome then 😂"

    return text


DANGLING_END_WORDS = {
    "usually", "mostly", "basically", "especially", "actually", "probably",
    "aur", "and", "ki", "ke", "ka", "ko", "ya", "but", "lekin", "agar",
    "kyunki", "because", "so", "jaise", "like", "toh", "to"
}


def is_dangling_line(line: str) -> bool:
    """Check if a line ends on an incomplete connective or is an orphaned stub."""
    clean = re.sub(r"[\U00010000-\U0010ffff.,!?\-—–]+", "", line).strip().lower()
    words = clean.split()
    if not words:
        return True
    if words[-1] in DANGLING_END_WORDS:
        return True
    # Orphaned pronoun/particle stub without verb (e.g. 'main toh', 'mera toh', 'waise main')
    if len(words) <= 2 and words[0] in {"main", "mera", "meri", "mere", "hum", "waise", "wese"}:
        if words[-1] in {"toh", "to", "bhi", "hi", "main", "mera"}:
            return True
    return False


def clean_and_assemble_lines(raw_lines: List[str]) -> List[str]:
    """
    Assembles lines by stitching mid-sentence breaks together and dropping
    orphaned dangling fragments that cannot be completed.
    """
    valid_lines: List[str] = []
    i = 0
    while i < len(raw_lines):
        line = raw_lines[i].strip()
        if not line:
            i += 1
            continue

        if is_dangling_line(line):
            if i + 1 < len(raw_lines):
                next_line = raw_lines[i + 1].strip()
                next_lower = next_line.lower()
                next_words = next_lower.split()
                first_word = next_words[0] if next_words else ""

                # If next line is a subject switch or standalone question, don't merge
                is_separate_thought = (
                    first_word in {"tum", "tumhara", "tumhare", "tumhari", "tumhe", "aap", "aapka", "kya", "waise", "wese", "kyun", "kahan", "kaise"}
                    or next_line.endswith("?")
                )
                if not is_separate_thought:
                    # Do not lowercase the first letter if the word is an acronym or already has uppercase letters (e.g. AWM, BGMI, AI, PUBG)
                    if len(next_line) > 1 and next_line[0].isupper() and not next_line[1].isupper():
                        merged_line = f"{line} {next_line[0].lower() + next_line[1:]}"
                    else:
                        merged_line = f"{line} {next_line}"
                    if not is_dangling_line(merged_line):
                        valid_lines.append(merged_line)
                    i += 2
                    continue
            # Drop incomplete dangling fragment
            i += 1
            continue

        valid_lines.append(line)
        i += 1

    return valid_lines


def arrange_and_format_response(
    raw_response: str,
    persona_name: str,
    recent_history: Optional[List[Dict[str, str]]] = None,
    user_gender: Optional[str] = None,
    preferred_pronoun: Optional[str] = "tum",
    user_message: Optional[str] = None,
    allow_emoji: bool = True,
    last_bot_emoji: Optional[str] = None,
    is_care_mode: bool = False,
    force_single_line: bool = False,
) -> str:
    """
    Arranges, styles, and formats the raw LLM response from OpenAI according to:
      1. User-specific context (preferred pronoun 'tum'/'aap', grammatical gender agreement).
      2. Persona character & natural mobile texting format (brevity, casing, emoji moderation).
      3. Clean-up of AI leaks, code fences, corporate assistant disclaimers, or metadata tags.
      4. Banned metaphor cleanup (removing "trailer mode", "negotiation", "budget", "vibe hijack", "araam mode").
      5. Strict Care Mode emoji and tone guard (no laughing/flirty emojis during sickness/distress).
      6. Preserving structural variety (allowing short 3-8 word single-line texts, lowercase, not forcing a 2-line split).
    """
    if is_user_in_distress(user_message):
        is_care_mode = True

    # 1. Base validation and cleaning with user context
    text = validate_and_clean_response(
        raw_response=raw_response,
        persona_name=persona_name,
        recent_history=recent_history,
        user_gender=user_gender,
        preferred_pronoun=preferred_pronoun,
        user_message=user_message,
    )

    # 1b. Clean banned metaphor phrases
    text = clean_banned_metaphors(text)

    # 2. Convert formal dashes and em-dashes (—, –) to space
    text = re.sub(r"\s*[—–]\s*", " ", text)
    text = re.sub(r"\s+-\s+", " ", text)
    # Convert formal semicolons to spaces
    text = text.replace(";", "")

    # 3. Clean formal bookish commas after conversational openers and before transition words
    text = re.sub(r"\b(haha|hehe|lol|achha|acha|arey|arre|waise|wese|yaar|yrr|suno|oye|haan|ha|hn|theek|thik|toh|to)\s*,\s*", r"\1 ", text, flags=re.I)
    text = re.sub(r",\s*(kya|toh|to|aur|par|lekin|waise|wese|yrr|yaar)\b", r" \1", text, flags=re.I)

    # Clean redundant spaces & punctuation combos
    text = re.sub(r"[ \t]+", " ", text).strip()
    text = re.sub(r",\s*,+", ", ", text)
    text = re.sub(r"\s*([.,!?])", r"\1", text)
    text = re.sub(r",\s*([?!])", r"\1", text)
    text = re.sub(r"\?{2,}", "?", text)
    text = re.sub(r"!{2,}", "!", text)

    # Moderate exclamation marks: Limit to at most 1 '!' in the entire reply
    if text.count("!") > 1:
        first_excl = text.find("!")
        before = text[:first_excl + 1]
        after = text[first_excl + 1:].replace("!", "")
        text = (before + after).strip()

    # 4. Emoji rules: at most 1 in 3 messages, no repeated emoji, no laughing emoji in care mode
    text = sanitize_emojis(
        text,
        allow_emoji=allow_emoji,
        last_bot_emoji=last_bot_emoji,
        is_care_mode=is_care_mode,
    )

    # 5. Structure variety & clean line breaks:
    # If the response is a single continuous string with multiple distinct sentences or questions,
    # split at sentence boundaries so distinct thoughts appear on new lines (WhatsApp style)
    if "\n" not in text and len(text.split()) > 7:
        split_pattern = r"(?<=[?!])\s+(?=[A-Za-z\u0900-\u097F])|(?<=[.।])\s+(?=[A-Z\u0900-\u097F])"
        sentence_parts = re.split(split_pattern, text)
        if len(sentence_parts) > 1:
            raw_lines = [re.sub(r",\s*$", "", l.strip()).rstrip(".") for l in sentence_parts if l.strip()]
        else:
            raw_lines = [re.sub(r",\s*$", "", l.strip()).rstrip(".") for l in text.split("\n") if l.strip()]
    else:
        raw_lines = [re.sub(r",\s*$", "", l.strip()).rstrip(".") for l in text.split("\n") if l.strip()]

    # Clean dangling fragments & stitch mid-sentence breaks into complete sentences
    valid_lines = clean_and_assemble_lines(raw_lines)
    if not valid_lines:
        from app.utils.language_detector import detect_language
        lang = detect_language(user_message or "")
        if lang == "hindi":
            return "अरे सच में? क्या हुआ यार"
        elif lang == "hinglish":
            return "kya hua yaar, sab theek?"
        else:
            return "what happened, all good?"

    # Only collapse to a single line if force_single_line was requested AND the text is genuinely very short (<= 6 words)
    if force_single_line and len(" ".join(valid_lines).split()) <= 6:
        return " ".join(valid_lines)

    return "\n".join(valid_lines[:3])
