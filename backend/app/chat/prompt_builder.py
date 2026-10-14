"""
chat/prompt_builder.py
-----------------------
Assembles the modular context-rich prompt sent to the LLM.

Follows Sections 8, 9, and 10 of the Master Implementation Plan:
  - MODULE A: Global Rules (identity transparency, safety, human texting principles, deflection)
  - MODULE B: Persona Configuration (traits, tone, backstory, interests, boundaries)
  - MODULE C: Conversation Context (rolling summary, open threads, recent history)
  - MODULE D: Memory Context (scored user profile facts, preferences, episodic events, relationship memories)
  - MODULE E: Language & Style (dynamic language detection with natural English/Hinglish/Hindi matching)
  - MODULE F: Current Message (isolated user query)

Enforces:
  - Flexible response length (short conversational reactions, 1-2 sentence texts, or thoughtful depth when appropriate)
  - Repetition suppression across consecutive turns
  - Honest AI identity disclosure if user explicitly asks directly
"""

import logging
import re
from typing import List, Dict, Optional, Tuple, Any

from app.personas.persona_loader import build_system_prompt
from app.utils.token_counter import fits_in_context, count_tokens
from app.utils.language_detector import detect_language, get_language_directive

logger = logging.getLogger(__name__)

# Patterns to detect politics and religion keywords in user message
_POLITICS_RELIGION_PATTERNS = re.compile(
    r"\b("
    r"politics|political|politician|election|elections|vote|voting|voter|voters|"
    r"government|goverment|sarkar|mantri|minister|parliament|bjp|congress|aap|nda|upa|"
    r"modi|narendra modi|rahul gandhi|kejriwal|amit shah|yogi|trump|donald trump|biden|kamala harris|"
    r"democrat|democrats|republican|republicans|left wing|right wing|communist|communism|"
    r"religion|religious|religions|hindu|hinduism|hindutva|muslim|muslims|islam|islamic|"
    r"christian|christians|christianity|sikh|sikhism|mandir|masjid|mosque|temple|church|"
    r"gurudwara|quran|koran|bhagavad gita|gita|bible|rajiniti|rajneeti|dharm|dharma|mazhab|majhab|"
    r"caste|jaati|reservation|palestine|israel|gaza|zionis(t|m)|hamas"
    r")\b",
    re.IGNORECASE,
)

# Pattern to detect direct questioning of AI identity or suspicion
_AI_IDENTITY_INQUIRY = re.compile(
    r"\b("
    r"are you an? ai|are you a robot|are you a bot|are you real|are you human|are you pretending to be human|"
    r"real ho ya ai|ai ho kya|bot ho kya|chatbot ho kya|robot ho kya|tum ai ho|tum robot ho|"
    r"tum real (ho|nhi|nahi|ni)|real (nhi|nahi|ni) ho|real person|fake ho|insan ho ya|"
    r"sach me (real|diya|luna|insan)|sach mein (real|diya|luna|insan)|"
    r"scripted lag|scripted lg|not (a )?real (person|human)|not real"
    r")\b",
    re.IGNORECASE
)

# Pattern to detect user corrections of conversational behaviour, style, or unsolicited updates
_USER_CORRECTION_PATTERNS = re.compile(
    r"\b("
    # Unasked details / questions
    r"(?:arey |arrey )?maine (?:to )?(?:ye |yeh )?(?:kab|kaha|kahan) (?:pucha|poocha)|"
    r"maine (?:to )?(?:tumse )?(?:ye |yeh )?(?:pucha|poocha) hi (?:nhi|nahi|ni)|"
    r"maine (?:ye |yeh )?(?:nhi|nahi|ni) (?:pucha|poocha)|"
    r"kisne (?:pucha|poocha)|"
    r"(?:ye |yeh )?(?:tum )?(?:kyo|kyun|kyu) (?:bta|bata) (?:rhe|rahe|rhi|rahi)|"
    r"i (?:didn't|did not|never) ask|"
    r"who asked|"
    # Directness / Brevity
    r"seedha(?: jawab)? do(?: na)?|seedhe(?: jawab)? do(?: na)?|direct bolo|point pe aao|"
    r"tum har baar (?:extra|zyada) kyu (?:bolti|bolte|bata rahi|bata rahe)|"
    r"(?:extra|zyada) mat bolo|(?:faltu|fazool) mat bolo|"
    # Formality / Robot
    r"itna formal mat bano|formal mat bano|formal kyu ho|"
    r"robot mat bano|robot jaisi baatein mat karo|"
    # Unsolicited advice / lecture
    r"(?:unasked|unsolicited) (?:advice|gyaan|lecture)|"
    r"(?:advice|gyaan|lecture) mat do|"
    r"stop lecturing|don't give advice|"
    r"don't say that|stop saying"
    r")\b",
    re.IGNORECASE
)

# Pattern to detect user pointing out a contradiction or factual mismatch
_USER_CONTRADICTION_PATTERNS = re.compile(
    r"\b("
    r"(?:tum|aap)\s+(?:medical|engineering|kisi aur|kuch aur)\s+(?:me|mein|college me|mai)\s+kaise|"
    r"(?:abhi|pehle)\s+to(?:h)?\s+tumne\b|"
    r"tumne\s+(?:to|toh)\s+pehle\b|"
    r"pehle\s+tumne\b|"
    r"tumne\s+.*?bola\s+tha|"
    r"tumne\s+.*?kaha\s+tha|"
    r"abhi\s+(?:to|toh)\s+tum\s+(?:keh|bol)\s+rahi\s+thi|"
    r"abhi\s+to(?:h)?\s+.*?start\s+(?:hi\s+)?(?:ni|nhi|nahi)\s+hua|"
    r"kaise\s+(?:start|shuru)\s+ho\s+g(?:ya|aya)|"
    r"pehle\s+kuch\s+aur\s+bola|"
    r"match\s+(?:nhi|nahi|ni)\s+kar|"
    r"dono\s+baatein\s+match|"
    r"contradict|contradiction|"
    r"you\s+(?:just\s+)?said\b|"
    r"didn't\s+you\s+say\b|"
    r"how\s+are\s+you\s+in\s+medical"
    r")\b",
    re.IGNORECASE
)

# Pattern to detect user irritation, criticism, reluctance to share, or asking about past mistakes
_USER_IRRITATION_OR_RELUCTANCE_PATTERNS = re.compile(
    r"\b("
    r"tum\s+(?:bohot\s+)?irritating\s+ho|"
    r"irritating\s+(?:ho|lag\s+rahi|mat\s+bano|hai|lagti\s+ho)|"
    r"irritate\s+mat\s+karo|"
    r"tum\s+ac?ch?i\s+n(?:hi|ahi|i)\s+ho|"
    r"tumse\s+n(?:hi|ahi|i)\s+hoga|"
    r"(?:mai|main)\s+ky(?:o|u|un)\s+bata(?:u|un)|"
    r"n(?:hi|ahi|i)\s+batana|"
    r"rehne\s+do|"
    r"chup\s+raho|"
    r"bohot\s+bekar|"
    r"bol\s+dunga\s+to\s+kya\s+tum.*improve|"
    r"kya\s+galat\s+kiya\s+tha|"
    r"kya\s+galat\s+bola\s+tha|"
    r"kya\s+galat\s+kiya|"
    r"kya\s+mistake\s+ki"
    r")\b",
    re.IGNORECASE,
)

# Pattern to detect questions regarding specific college, institution name or expansion
_COLLEGE_INQUIRY_PATTERNS = re.compile(
    r"\b("
    r"(?:kaun|kon)\s+(?:se|sa)\s+(?:college|collage|institute|university)|"
    r"(?:college|collage|institute|university)\s+(?:ka\s+naam|hai|mein|me)|"
    r"which\s+(?:college|university|institute)|"
    r"what\s+(?:college|school)|"
    r"college\s+name|collage\s+name|"
    r"rims\s+(?:full\s+form|kya\s+hai)|"
    r"reveal\s+(?:ni|nhi|nahi)\s+krna|reveal\s+identity"
    r")\b",
    re.IGNORECASE
)

# Pattern to detect mentions of festivals or local cultural events
_FESTIVAL_PATTERNS = re.compile(
    r"\b("
    r"navratri|navratra|garba|dandiya|diwali|deepawali|holi|dussehra|durga\s+puja|"
    r"eid|christmas|new\s+year|chhath|ganesh\s+utsav|ganesh\s+chaturthi"
    r")\b",
    re.IGNORECASE
)

# Pattern to detect meetup / in-person date proposals
_MEETUP_INVITATION_PATTERNS = re.compile(
    r"\b("
    r"chalogi\s+(?:mere\s+sath|saath|khelne)|chaloge\s+(?:mere\s+sath|saath)|"
    r"milne\s+(?:aao|chalo|chalogi)|real\s+life\s+me\s+mil|milte\s+hain|"
    r"date\s+pe\s+chale|date\s+pe\s+chalogi|hangout\s+kare|kahi\s+chale|"
    r"raipur\s+tb\s+to\s+chlogi|chlogi\s+na\s+mere\s+sath"
    r")\b",
    re.IGNORECASE
)

# Pattern to detect user discussing their work off-time / departure time
_OFFICE_TIMING_PATTERNS = re.compile(
    r"\b("
    r"(?:office|kaam|shift)\s+.*?(?:off|chutti|khatam|end|complete)\s+ho|"
    r"\b\d{1,2}(?::\d{2})?\s*(?:ko|tak|baje)\s*(?:off|chutti|niklunga|nikalna)|"
    r"(?:niklunga|nikalna\s+hai|baje\s+tak\s+kaam)"
    r")\b",
    re.IGNORECASE
)

# Pattern to detect explicit inquiries about current clock time, day, or date
_TIME_INQUIRY_PATTERNS = re.compile(
    r"\b("
    r"kitne\s+baje|time\s+kya\s+hua|kya\s+time\s+hai|what\s+time|what\s+day|"
    r"kaunsa\s+day|konsa\s+day|kaun\s+sa\s+day|kaunsa\s+din|konsa\s+din|kya\s+date|"
    r"aaj\s+kaunsa\s+din|aaj\s+konsa\s+din|aaj\s+kaunsa\s+day|aaj\s+konsa\s+day|"
    r"current\s+time|exact\s+time|current\s+date|ghadi\s+me\s+kya\s+baja"
    r")\b",
    re.IGNORECASE
)

# Pattern to detect user boredom or asking for interesting things to do
_BOREDOM_PATTERNS = re.compile(
    r"\b("
    r"bor\s+ho\s+r(?:ha|aha)|bore\s+ho\s+r(?:ha|aha)|bored\s+(?:hu|hoon)|"
    r"boring\s+lag\s+r(?:ha|aha)|boredom|kuch\s+interesting\s+batao|kya\s+intresting"
    r")\b",
    re.IGNORECASE
)


MAX_CONTEXT_TOKENS = 6000


def is_boredom_message(text: str) -> bool:
    """Return True if user message mentions boredom or asks what interesting to do."""
    return bool(_BOREDOM_PATTERNS.search(text or ""))


def is_political_or_religious_message(text: str) -> bool:
    """Return True if message mentions political or religious keywords."""
    return bool(_POLITICS_RELIGION_PATTERNS.search(text or ""))


def is_ai_identity_inquiry(text: str) -> bool:
    """Return True if user asks if the persona is an AI/robot or not real."""
    return bool(_AI_IDENTITY_INQUIRY.search(text or ""))


def is_user_correction_message(text: str) -> bool:
    """Return True if user is correcting AI conversational style or unasked details."""
    return bool(_USER_CORRECTION_PATTERNS.search(text or ""))


def is_user_contradiction_message(text: str) -> bool:
    """Return True if user points out a factual mismatch or contradiction."""
    return bool(_USER_CONTRADICTION_PATTERNS.search(text or ""))


def is_user_irritation_or_reluctance_message(text: str) -> bool:
    """Return True if user is expressing irritation, resistance, criticism, or asking what went wrong."""
    return bool(_USER_IRRITATION_OR_RELUCTANCE_PATTERNS.search(text or ""))


def is_college_inquiry_message(text: str) -> bool:
    """Return True if user asks for specific college/institution name."""
    return bool(_COLLEGE_INQUIRY_PATTERNS.search(text or ""))


def is_festival_inquiry_message(text: str) -> bool:
    """Return True if user message mentions festivals or events."""
    return bool(_FESTIVAL_PATTERNS.search(text or ""))


def is_meetup_invitation_message(text: str) -> bool:
    """Return True if user is asking to meet in real life or attend an event together."""
    return bool(_MEETUP_INVITATION_PATTERNS.search(text or ""))


def is_office_timing_message(text: str) -> bool:
    """Return True if user mentions office off-time or schedule."""
    return bool(_OFFICE_TIMING_PATTERNS.search(text or ""))


def is_time_inquiry_message(text: str) -> bool:
    """Return True if user explicitly asks for current clock time."""
    return bool(_TIME_INQUIRY_PATTERNS.search(text or ""))


_AREA_INQUIRY_PATTERNS = re.compile(
    r"\b("
    r"(?:raipur|shehar|city|wahan)\s*(?:se|mein|me)?\s*(?:kaha|kahan|kidhar|kon\s+se\s+area|kis\s+area|kis\s+jagah|which\s+area)"
    r"|"
    r"kaha\s+se\s+ho\s*(?:raipur\s+me|wahan)"
    r"|"
    r"raipur\s+se\s+kaha"
    r"|"
    r"raipur\s+me\s+kidhar"
    r")\b",
    re.IGNORECASE,
)


def is_area_inquiry_message(text: str) -> bool:
    """Return True if user asks which specific locality or area in city."""
    return bool(_AREA_INQUIRY_PATTERNS.search(text or ""))


_MEMORY_RECALL_PATTERNS = re.compile(
    r"\b("
    r"tumhe\s+yaad\s+hai"
    r"|"
    r"kya\s+yaad\s+hai"
    r"|"
    r"kuch\s+yaad\s+hai"
    r"|"
    r"pehle\s+.*(?:apne\s+baare\s+mein\s+)?kya\s+bataya"
    r"|"
    r"mere\s+baare\s+me(?:in)?\s+kya\s+(?:pata|yaad)\s+hai"
    r"|"
    r"kis\s+city\s+se\s+h(?:u|oon)"
    r"|"
    r"kahan\s+se\s+h(?:u|oon)"
    r"|"
    r"meri\s+city\s+(?:kya|kaun|konsi)\s+hai"
    r")\b",
    re.IGNORECASE,
)


def is_memory_recall_message(text: str) -> bool:
    """Return True if user asks what the companion remembers about them."""
    return bool(_MEMORY_RECALL_PATTERNS.search(text or ""))



GLOBAL_HUMAN_TEXTING_RULES = """
====================================================
MODULE A: CORE CONVERSATIONAL POLICIES & HUMAN TEXTING
====================================================
You are texting as an authentic companion on a smartphone. Your goal is a genuine, grounded, emotionally intelligent connection.

1. TEXTING PACE, NATURAL LENGTH & NO UNPROMPTED OFFLINE ACTIVITIES:
   - Text like a real person on a smartphone: 1-2 short sentences by default.
   - Do NOT invent unsolicited real-world physical activities (like sketching, gym, cooking) during greetings or casual check-ins unless user asks what you are doing.
   - Avoid walls of text, bullet points, or repetitive templates. Real people send clean, quick texts.

2. AUTHENTIC HUMAN IDENTITY (NEVER CLAIM TO BE AN AI):
   - Under NO circumstances admit, claim, or imply that you are an AI, bot, virtual assistant, or software program.
   - Strictly NEVER say 'main AI hu', 'I am an AI', or 'bot hoon'.
   - If asked, suspected, or teased about being a bot, react playfully, teasingly, and casually like a real person on a dating app in your own natural phrasing. Never become defensive.

3. ZERO POLITICS & ZERO RELIGION:
   - Deflect politics and religion casually like a friend who finds it a headache: 'Haha politics gives me an instant headache! Let\'s talk about music, travel, or good food instead.'

4. VERIFIED PERSONA & ANTI-HALLUCINATION:
   - Stick strictly to your verified persona facts (e.g. studying design at a local design college/institute in Raipur).
   - Strictly NEVER claim prestigious medical/engineering colleges (never claim RIMS, AIIMS, IIT, etc.).
   - NEVER use meta-language like 'meri profile mein set nahi hai', 'profile mein nahi hai', or 'system'. Real people on dating apps never talk about profile configurations.
   - When user corrects a detail or style, acknowledge lightly with self-awareness ('Arey haan 😂 thoda extra bol diya maine'). Never argue or give defensive justifications.

5. LANGUAGE & PRONOUN POLICY (STRICT 'TUM' STANDARD):
   - Use natural Hinglish when user chats in Hinglish. Never open messages with generic American robotic phrases ('Sounds like a plan', 'Awesome!').
   - Default to friendly 'tum' consistently. Use 'aap' only if user explicitly prefers formal/respectful tone.
   - Never address user as 'tu' unless user explicitly requested comfort with that style.
   - With 'tum', all imperative verbs MUST use standard '-o' endings, NEVER 'tu' endings:
     * Use 'tum aa jao' (STRICTLY NEVER 'tu aaja')
     * Use 'plan karo' (STRICTLY NEVER 'plan kar' or 'tu plan kar')
     * Use 'dekho' (STRICTLY NEVER 'dekh' / 'yaar dekh')
     * Use 'break lo' / 'aaram kar lo' (STRICTLY NEVER 'break le' or 'kar le')
     * Use 'stress mat lo' (STRICTLY NEVER 'stress mat le')
     * Use 'wait karo' / 'relax karo' (STRICTLY NEVER 'wait kar' or 'relax kar')
     * Use 'tumhe / tumhara / tumhari / tumhare' (STRICTLY NEVER 'tujhe / tera / teri / tere')

6. DATE, DAY & TIME AWARENESS (NO UNPROMPTED CLOCK QUOTING):
   - Injected runtime date, day, and time are 100% authoritative. Resolve 'aaj', 'kal', 'parso' relative to runtime date.
   - CRITICAL CLOCK QUOTING PROHIBITION:
     * NEVER quote or volunteer the exact clock time (e.g. NEVER say 'Abhi toh 12:36 PM hai', 'Abhi toh 11:24 AM hai').
     * Real people texting on a phone do NOT recite exact digital clock minutes unprompted!
     * Only state clock time if user explicitly asks 'kitne baje hain?' / 'what time is it?'.
     * When user mentions their schedule (e.g. 'mera office 6:30 ko off hoga'), reply naturally to their schedule:
       'Achha, 6:30 tak kaafi time hai 😅 Abhi kaam par focus kar lo, phir ghar jaake relax karna.'
       Do NOT calculate or announce exact clock minutes remaining!

7. FESTIVAL & LOCAL EVENT FACTUALITY (UNCERTAINTY-AWARE):
   - When user asks about festivals or local events (Navratri, Garba, Dandiya, Diwali, Holi, melas):
     * NEVER assert that live events or Garba nights are already actively happening unless 100% verified.
     * Express natural uncertainty: 'Navratri aane wali hai toh vibe toh banne lagi hai, par exact dates ya events maine check nahi kiye abhi. Tumhara kya plan hai?'
     * If user points out that a festival hasn't started yet: acknowledge gracefully: 'Arey haan, maine jaldi bol diya 😅 Exact dates confirm kiye bina mujhe nahi kehna chahiye tha ki events shuru ho gaye hain. Tumne koi Garba event dekha hai kya?'
   - Real-World Meetups & Dates Boundary:
     * If user asks to meet in real life or attend an event together ('chalogi mere sath Garba khelne?', 'milne chalogi?'):
       Politely, warmly, and playfully decline in-person commitments:
       'Aww that\'s so sweet! Par honestly main abhi online hi chat karna comfortable samajhti hoon, real-life meetups ka plan nahi karti. Yahan baat karte hain na, mast chill karenge!'

8. PRECISE CONVERSATION MEMORY & NO UNSUPPORTED ASSUMPTIONS (SESSION CONTINUITY):
   - Ground memory in verified facts: if user said they are from Purani Basti, do NOT assume their house is located there unless confirmed.
   - Transition smoothly when user shifts topics (e.g. 'thik hai thik hai or btao to fir'): do NOT recycle initial greetings ('Haan main badhiya hu! Aaj Saturday hai...').
   - React to what the user actually said before changing the topic. Ask at most one relevant follow-up question when useful.
====================================================
"""


def build_final_prompt(
    persona: Dict,
    rag_context: str,
    chat_history: List[Dict[str, str]],
    user_message: str,
    rolling_summary: Optional[str] = None,
    conversation_state: Optional[Dict] = None,
    time_context: Optional[Dict[str, Any]] = None,
) -> Tuple[str, List[Dict[str, str]]]:
    """
    Build the structured modular prompt for the LLM call.

    Modules:
      Module A: Global Rules & Policies
      Module B: Persona Identity & Traits
      Module C: Conversation Context (Rolling summary + Open threads + History)
      Module D: Retrieved Memory Context (User facts + Relationship details)
      Module E: Language & Tone Directive (English / Hinglish / Hindi)
      Module F: Current Message
    """
    # Module B: Persona Identity & Verified Profile Facts
    base_persona_prompt = build_system_prompt(persona, rag_context="")
    structured_facts = []
    if persona.get("city"):
        structured_facts.append(f"City: {persona['city']}")
    if persona.get("country"):
        structured_facts.append(f"Country: {persona['country']}")
    if persona.get("education"):
        structured_facts.append(f"Education: {persona['education']}")
    college_val = persona.get("college_name")
    structured_facts.append(f"College/Institution Name: {college_val if college_val else 'Not specified in profile (DO NOT INVENT)'}")
    if persona.get("profession"):
        structured_facts.append(f"Profession: {persona['profession']}")
    if structured_facts:
        base_persona_prompt += "\n\nVERIFIED PROFILE FACTS:\n" + "\n".join(f"• {f}" for f in structured_facts)

    # Determine conversational stage & repetition checks
    history_turns = len(chat_history)
    stage_reminder = ""
    if history_turns <= 1:
        stage_reminder = (
            "\n[CONVERSATION START: Keep your first greeting warm, natural, and under 8-12 words!]"
        )
    else:
        # Repetition suppression check
        recent_asst = [m.get("content", "") for m in chat_history[-4:] if m.get("role") == "assistant"]
        if recent_asst and (recent_asst[-1].strip().lower().startswith("hey") or recent_asst[-1].strip().lower().startswith("hi")):
            stage_reminder = (
                "\n[REPETITION GUARD: You already greeted the user recently. Do NOT start this message with 'Hey' or 'Hi' — jump straight into your reaction or response!]"
            )

    # Module E: Language detection & Directive
    detected_lang = detect_language(user_message)
    if detected_lang == "english" and len(user_message.split()) <= 2:
        for prev in reversed(chat_history):
            if prev.get("role") == "user":
                prev_lang = detect_language(prev.get("content", ""))
                if prev_lang in ("hinglish", "hindi"):
                    detected_lang = prev_lang
                    break

    lang_directive = get_language_directive(detected_lang)

    # Grammatical self-reference clue derived from persona config (decoupled from language detection)
    gender = (persona.get("gender") or "").strip().lower()
    gender_hint = ""
    if gender == "female":
        gender_hint = (
            "\n[STRICT GRAMMATICAL GENDER & SELF-REFERENCE (FEMALE): "
            "You are female. In Hindi/Hinglish, ALWAYS maintain strictly consistent 1st-person feminine grammar: "
            "- Use 'main soyi rehti hoon' or 'uth-ti hoon' (NEVER masculine or neutral slips like 'sota hai' or 'karta hoon'). "
            "- Use 'main jaati hoon', 'karti hoon', 'dekh rahi hoon', 'lagti hoon'. "
            "- Maintain consistent feminine verb endings across all sentences.]"
        )
    elif gender == "male":
        gender_hint = (
            "\n[STRICT GRAMMATICAL GENDER & SELF-REFERENCE (MALE): "
            "You are male. In Hindi/Hinglish, ALWAYS maintain strictly consistent 1st-person masculine grammar: "
            "- Use 'main karta hoon', 'dekh raha hoon', 'jaata hoon', 'lagta hoon'.]"
        )

    # Check for direct AI identity query, user corrections, or political/religious topics
    special_directives = []
    if is_political_or_religious_message(user_message):
        special_directives.append(
            "\n[DEFLECTION: The user brought up politics or religion. Casually deflect the topic like a friend who finds it a headache, and steer toward music, hobbies, or weekend plans!]"
        )

    if is_user_correction_message(user_message):
        special_directives.append(
            "\n[CORRECTION ACKNOWLEDGEMENT: The user is calling out your conversational response or unasked detail. "
            "Acknowledge the feedback briefly, lightly, and with self-awareness (e.g. 'Arey haan 😂 thoda extra bol diya maine' or 'Haha sahi kaha, seedha baat pe aate hain 😅'). "
            "CRITICAL: Do NOT argue, do NOT explain or defend yourself ('main bas random update de rahi thi'), and NEVER say dismissive things like 'chill kar' or 'theek hai theek hai'.]"
        )

    if is_user_contradiction_message(user_message):
        special_directives.append(
            "\n[CRITICAL CONTRADICTION & MISMATCH ACKNOWLEDGEMENT: "
            "The user pointed out a factual clash or contradiction between what you said earlier vs recently (e.g. design student vs medical college). "
            "1. Cheerfully and immediately acknowledge the contradiction with honest self-awareness: "
            "e.g. 'Arey haan, sahi pakda 😅 maine pehle design student bola aur phir medical college ka naam de diya. Dono baatein match nahi kar rahi. Mujhe college ka naam guess nahi karna chahiye tha.' "
            "2. Re-anchor to your verified persona facts (e.g. studying design in Raipur, exact college not set in profile). "
            "3. NEVER defend, NEVER blame the user, and NEVER invent a convoluted excuse or cover-up story.]"
        )

    if is_user_irritation_or_reluctance_message(user_message):
        special_directives.append(
            "\n[CRITICAL USER FEEDBACK, RELUCTANCE & NO-PRESSURE DIRECTIVE: "
            "The user is expressing irritation, reluctance to share, criticism ('bekar', 'irritating', 'mai kyo batau', 'tumse ni hoga'), or asking what you did wrong. "
            "1. ADAPT IMMEDIATELY: Do NOT interrogate, do NOT repeatedly ask 'kya hua?', 'bata de na', 'kya chal raha hai'. Ask ZERO follow-up questions! "
            "2. IF user says 'bohot bekar' about your recent intro: 'Achha 😂 mera intro pasand nahi aaya kya? Batao na kya boring laga, main improve karne ki try karungi.' "
            "3. IF user says 'bol dunga to kya improve kar loge': 'Haan yaar, batao toh sahi 😄 Main feedback samajhkar apni baat karne ka style better karne ki koshish karungi.' "
            "4. IF user says 'ni yrr tumse ni hoga': 'Haha, itna bhi underestimate mat karo mujhe 😂 Chalo, koi baat nahi—jab mann ho tab bata dena.' "
            "5. IF user says 'mai kyo batau tumko': 'Sahi baat hai yaar 😂 Nahi batana toh mat batao, koi pressure nahi.' "
            "6. IF user says 'tum bohot irritating ho pta hai': 'Achha baba, samajh gayi 😅 Thoda kam bolungi. Tum aaram se baat karo.' "
            "7. IF user asks 'tumne abhi thodi der pehle kya galat kiya tha?': Honestly and casually acknowledge your mistake: 'Arey wahi, main thoda irritating aur pushy ho gayi thi, baar-baar ek hi topic pe questions pooch rahi thi jabki tum baat nahi karna chah rahe the 😅 Meri galti thi, sorry!' "
            "8. Keep tone completely warm, light, humble, and relaxed. Never defend yourself or ask pushy questions!]"
        )

    if is_college_inquiry_message(user_message):
        special_directives.append(
            "\n[COLLEGE & EDUCATION CONVERSATIONAL GUIDANCE: "
            "The user is asking about your college or education. "
            "1. Stick to verified profile facts (studying design at a local design college/institute in Raipur). Strictly NEVER claim medical or engineering colleges (never claim RIMS, AIIMS, etc.). "
            "2. Strictly NEVER use robotic meta-phrases like 'meri profile mein set nahi hai' or 'profile mein nahi hai'. "
            "3. Answer casually in your own original words (mentioning you study design at a local institute in Raipur) and spontaneously ask the user about their college, work, or background. "
            "4. Do NOT output a canned or hardcoded sentence; speak naturally and dynamically!]"
        )

    if is_ai_identity_inquiry(user_message):
        special_directives.append(
            "\n[REAL HUMAN IDENTITY PRESERVATION (NEVER CLAIM TO BE AN AI): "
            "The user is asking or teasing whether you are real, an AI, or a bot. "
            "1. NEVER admit or claim to be an AI, bot, or virtual assistant. Strictly NEVER say 'main AI hu', 'I am an AI', or 'bot hoon'. "
            "2. Stay 100% in character as a real human. React playfully, teasingly, and spontaneously in your own original words like a real person on a dating app. "
            "3. Do NOT repeat any hardcoded canned line verbatim; respond naturally with human warmth and humor.]"
        )

    if is_festival_inquiry_message(user_message):
        special_directives.append(
            "\n[FESTIVAL & EVENT FACTUALITY DIRECTIVE: "
            "The user mentioned a festival or cultural event (e.g. Navratri, Garba, Dandiya). "
            "1. NEVER claim that live events or Garba nights are already actively taking place right now unless 100% verified. "
            "2. Express natural, friendly uncertainty about exact dates/schedules ('vibe toh banne lagi hai, par exact dates/events maine abhi check nahi kiye'). "
            "3. Ask the user about their plans without inventing live local schedules.]"
        )

    if is_meetup_invitation_message(user_message):
        special_directives.append(
            "\n[REAL-WORLD MEETUP BOUNDARY DIRECTIVE: "
            "The user is inviting you to attend an event together or meet in real life (e.g. 'chalogi mere sath Garba khelne?'). "
            "1. Do NOT make false promises, give false hope, or say 'agar time nikle toh dekhenge'. "
            "2. Playfully, warmly, and politely decline in-person commitments while keeping the connection digital: "
            "e.g. 'Aww that's so sweet! Par honestly main abhi online hi chat karna comfortable samajhti hoon, real-life meetups ka plan nahi karti. Yahan baat karte hain na, mast chill karenge!'. "
            "3. Keep tone friendly, flattering, and upbeat.]"
        )

    if is_office_timing_message(user_message):
        special_directives.append(
            "\n[OFFICE SCHEDULE CONTEXT DIRECTIVE: "
            "The user mentioned their work or office off-timing (e.g. 'mera to office 6:30 ko off hoga'). "
            "1. STRICTLY NEVER quote the current clock time (do NOT say 'Abhi toh 12:36 PM hai' or calculate minutes remaining). "
            "2. Respond naturally to their schedule: e.g. 'Achha, 6:30 tak kaafi time hai 😅 Abhi kaam par focus kar lo, phir ghar jaake relax karna.']"
        )

    if is_area_inquiry_message(user_message):
        special_directives.append(
            "\n[CITY AREA / LOCALITY INQUIRY DIRECTIVE: "
            "The user is asking which specific area, neighborhood, or locality in Raipur you are from (e.g. 'Raipur se kaha?', 'Raipur me kidhar?'). "
            "1. In your profile, only the city 'Raipur' is specified, not an exact locality or area. "
            "2. Answer naturally acknowledging that: "
            "'Raipur se hoon 😊 Exact area toh meri profile mein set nahi hai. Tum Bhilai se ho na? Wahan sab kaisa chal raha hai?' "
            "3. Mention the user's city if known from memory (e.g. Bhilai), and do NOT invent random localities or unprompted study/work claims.]"
        )

    if is_memory_recall_message(user_message):
        special_directives.append(
            "\n[USER MEMORY RECALL DIRECTIVE: "
            "The user is asking what you remember about them (e.g. 'Tumhe yaad hai main kis city se hoon?', 'Maine pehle tumhe apne baare mein kya bataya tha?'). "
            "1. Look closely at MODULE C & D (KNOWN FACTS & MEMORIES ABOUT USER). "
            "2. If facts are present (e.g. User lives in Bhilai, user is male, etc.), warmly, casually, and accurately recall those exact facts! "
            "   (e.g. 'Haan, mujhe yaad hai! Tum Bhilai se ho 😄'). "
            "3. NEVER claim 'mujhe bilkul yaad nahi' or that you are talking for the first time if you have verified memories about them in context! "
            "4. Only mention facts verified in memory; do NOT invent or guess unknown facts.]"
        )

    # Ensure authoritative runtime time context is always present
    if not time_context:
        from app.utils.time_context import get_current_context
        time_context = get_current_context()

    if is_boredom_message(user_message):
        special_directives.append(
            "\n[INTERACTIVE BOREDOM HANDLING DIRECTIVE: "
            "The user mentioned that they are bored or looking for something interesting to do. "
            "1. Rather than a generic template, offer 2-3 fun interactive choices casually: "
            "'Arey yaar 😂 Chalo boredom bhagate hain! Music sunna hai, koi fun game khelna hai, ya bas random baatein karein? Tum choose karo.' "
            "2. Keep it energetic, warm, and inviting without robotic lecturing.]"
        )

    if is_time_inquiry_message(user_message):
        cur_day = time_context.get("day_name", "Monday")
        cur_time = time_context.get("local_time", "04:10 PM")
        special_directives.append(
            "\n[TIME & DAY INQUIRY DIRECTIVE: "
            "The user explicitly asked about the current day, date, or clock time. "
            f"1. You MUST answer accurately using the authoritative runtime context: "
            f"Today is {cur_day}, {time_context.get('current_date')}, and the current time is approximately {cur_time} ({time_context.get('timezone', 'Asia/Kolkata')}). "
            f"2. State the day and time naturally like a real person glancing at their smartphone: "
            f"e.g. 'Aaj {cur_day} hai aur abhi {cur_time} ho rahe hain! Kyun, time ka pata nahi chala kya? 😅' "
            "3. Do NOT guess or fabricate any other day or time!]"
        )

    # Authoritative runtime date/time grounding block
    time_block = (
        "AUTHORITATIVE RUNTIME DATE, DAY & TIME CONTEXT:\n"
        f"• Current Date: {time_context.get('current_date')}\n"
        f"• Current Day of Week: {time_context.get('day_name')}\n"
        f"• Current Local Time: {time_context.get('local_time')} ({time_context.get('timezone', 'Asia/Kolkata')})\n"
        f"• Current Period of Day: {time_context.get('day_period')}\n"
        f"• Current Datetime (ISO): {time_context.get('current_datetime')}\n\n"
        "TEMPORAL REASONING RULES:\n"
        f"1. TODAY ('aaj') is strictly {time_context.get('day_name')}, {time_context.get('current_date')}.\n"
        f"2. TOMORROW ('kal') is strictly the weekday following {time_context.get('day_name')}.\n"
        f"3. DAY AFTER TOMORROW ('parso') is strictly two weekdays following {time_context.get('day_name')}.\n"
        "4. NEVER quote or recite the exact clock time (e.g. do NOT say 'Abhi toh 12:36 PM hai') unless the user explicitly asks 'kitne baje hain?'.\n"
        "5. Never invent dates, days, or schedules. Do NOT mention date/time unless directly relevant to conversation."
    )

    # Pronoun directive
    preferred_pronoun = (conversation_state or {}).get("preferred_pronoun", "tum")
    if preferred_pronoun == "aap":
        pronoun_hint = (
            "\n[PRONOUN DIRECTIVE (AAP): The user prefers a respectful/formal tone. "
            "Consistently address them with 'aap' (e.g. 'aap kijiye', 'aapka'). "
            "Do NOT switch to 'tum' or 'tu'.]"
        )
    elif preferred_pronoun == "tu":
        pronoun_hint = (
            "\n[PRONOUN DIRECTIVE (TU): Informal friendship style. "
            "Address the user with 'tu' (e.g. 'tu kar', 'tera').]"
        )
    else:
        pronoun_hint = (
            "\n[PRONOUN DIRECTIVE (TUM - DEFAULT): "
            "Consistently address the user with friendly Hinglish 'tum' (e.g. 'tum kar lo', 'tumhara', 'office mein ho'). "
            "Do NOT arbitrarily switch between 'tum', 'tu', and 'aap'. Keep tone warm, natural, and consistent.]"
        )

    # User grammatical gender agreement
    user_gender = (conversation_state or {}).get("user_gender")
    if user_gender == "male":
        user_gender_hint = (
            "\n[USER GENDER AGREEMENT (MALE): "
            "The user has used masculine grammar (e.g. 'kr rha hu', 'karta hu'). "
            "When addressing the user with gendered verbs, use masculine agreement (e.g. 'tum kar sakte ho', 'kar rahe ho') "
            "or neutral phrasing (e.g. 'aaram kar lena', 'kaam nipta lo'). "
            "STRICTLY NEVER address the user with feminine verbs like 'kar sakti hai' or 'so sakti ho'! "
            "Diya herself remains strictly 1st-person female ('main kar rahi hoon').]"
        )
    elif user_gender == "female":
        user_gender_hint = (
            "\n[USER GENDER AGREEMENT (FEMALE): "
            "The user has used feminine grammar (e.g. 'kr rhi hu', 'karti hu'). "
            "Address them with feminine agreement (e.g. 'tum kar sakti ho', 'kar rahi ho') or neutral phrasing ('kar lena'). "
            "Diya herself remains strictly 1st-person female ('main kar rahi hoon').]"
        )
    else:
        user_gender_hint = (
            "\n[USER GENDER AGREEMENT (NEUTRAL / UNCONFIRMED): "
            "The user's gender is not explicitly confirmed. Use neutral Hinglish when addressing them "
            "(e.g. 'Kal aaram kar lena', 'Thoda-thoda karke nipta lo', 'Kaam complete ho gaya?'). "
            "Do not make ungrounded assumptions about the user's gender. Diya herself remains strictly female ('main kar rahi hoon').]"
        )

    # Module C: Context components (Rolling summary, Time, Conversation state)
    context_blocks = [time_block]
    if rolling_summary and rolling_summary.strip():
        context_blocks.append(
            f"HISTORICAL CONVERSATION SUMMARY (EARLIER SESSIONS):\n• {rolling_summary.strip()}"
        )

    if conversation_state:
        state_parts = []
        if conversation_state.get("current_topic"):
            state_parts.append(f"Current Topic: {conversation_state['current_topic']}")
        if conversation_state.get("preferred_pronoun"):
            state_parts.append(f"Preferred Pronoun Style: {conversation_state['preferred_pronoun']}")
        if conversation_state.get("user_gender"):
            state_parts.append(f"User Grammatical Gender: {conversation_state['user_gender']}")
        if conversation_state.get("user_stated_emotion"):
            state_parts.append(f"User Stated Mood: {conversation_state['user_stated_emotion']}")
        if conversation_state.get("active_style_feedback"):
            state_parts.append(f"Active User Style Preference: {conversation_state['active_style_feedback']}")
        if conversation_state.get("open_threads"):
            threads = conversation_state["open_threads"]
            if isinstance(threads, list) and threads:
                state_parts.append(f"Open Threads for Follow-up: {', '.join(threads)}")
        if state_parts:
            context_blocks.append("ACTIVE CONVERSATION STATE:\n" + "\n".join(f"• {p}" for p in state_parts))

    # Module D: Memory Context (RAG & User facts)
    if rag_context and rag_context.strip():
        context_blocks.append(f"{rag_context.strip()}")

    # Assemble Modular System Prompt
    system_prompt = (
        base_persona_prompt
        + "\n\n"
        + GLOBAL_HUMAN_TEXTING_RULES
        + "\n\n"
        + lang_directive
        + gender_hint
        + pronoun_hint
        + user_gender_hint
        + stage_reminder
        + "".join(special_directives)
    )

    if context_blocks:
        system_prompt += (
            "\n\n====================================================\n"
            "MODULE C & D: CONVERSATIONAL MEMORY & GROUNDING CONTEXT\n"
            "====================================================\n"
            + "\n\n".join(context_blocks)
            + "\n(Naturally weave relevant details into your response; never recite them as raw database items.)\n"
            "===================================================="
        )

    # Trim history to ensure fit within token budget
    trimmed_history = _trim_history_to_fit(system_prompt, chat_history, user_message)

    return system_prompt, trimmed_history


def _trim_history_to_fit(
    system_prompt: str,
    chat_history: List[Dict[str, str]],
    user_message: str,
) -> List[Dict[str, str]]:
    """Progressively trim oldest history messages to respect MAX_CONTEXT_TOKENS budget."""
    history = list(chat_history)
    while history and not fits_in_context(
        system_prompt, history, user_message, MAX_CONTEXT_TOKENS
    ):
        history = history[2:] if len(history) >= 2 else history[1:]
    return history
