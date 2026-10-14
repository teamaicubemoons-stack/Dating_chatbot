"""
memory/memory_extractor.py
--------------------------
Extracts, validates, deduplicates, and persists long-term candidate memories
from conversation exchanges.

Follows Section 6 of the Master Implementation Plan:
  1. Identify candidate memories across categories (profile facts, preferences,
     episodic events, relationship details, user corrections, emotional cues).
  2. Validate structured candidate attributes (type, confidence, importance, normalized key).
  3. Compare against existing memories to resolve duplicates and contradictions.
  4. Persist confirmed memories to SQLite (user_memories) and ChromaDB vector store.
  5. Update conversation state (current_topic, open_threads, user_stated_emotion).
  6. Non-blocking & fault-tolerant: failure never breaks the user's chat response.
"""

import logging
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from sqlalchemy.orm import Session

from app.memory.memory_store import (
    save_user_memory, get_active_user_memories, update_conversation_state,
    get_or_create_conversation_state
)
from app.rag.retriever import store_memory_fact
from app.models.db_models import UserMemory

logger = logging.getLogger(__name__)


def detect_pronoun_preference(text: str) -> Optional[str]:
    """
    Detect explicit user pronoun preference.
    Returns: 'tum' | 'aap' | 'tu' | None
    """
    t = (text or "").lower()
    # Preference for 'tum' (default friendly style, or rejecting aap / tu)
    if re.search(r'\b(?:mujhse\s+)?tum\s+(?:bolna|bolo|kaho|bola\s+karo|hi\s+bolo)\b', t) or \
       re.search(r'\b(?:aap|tu)\s+mat\s+(?:bolna|bolo|kaho|use\s+karo)\b', t):
        return 'tum'
    # Preference for 'aap' (formal / respectful)
    if re.search(r'\b(?:mujhse\s+)?aap\s+(?:bolna|bolo|kaho|boliye|bola\s+karo)\b', t):
        return 'aap'
    # Preference for 'tu' (close informal)
    if re.search(r'\b(?:mujhse\s+)?tu\s+(?:bolna|bolo|bol|kaho|bola\s+kar)\b', t):
        return 'tu'
    return None


def detect_user_gender_cue(text: str) -> Optional[str]:
    """
    Detect user's 1st-person grammatical gender cues from Hinglish/Hindi or English.
    Returns: 'male' | 'female' | None
    """
    t = (text or "").lower()
    # 1st person masculine verb endings: kr rha hu, kar raha hoon, karta hu, etc.
    if re.search(r'\b(?:kr|kar|ja|dekh|baitha|padh|soch|aa|khel|khaa|ruk|le|de)\s*(?:raha|rha)\s+(?:hu|hoon|hun)\b', t) or \
       re.search(r'\b(?:karta|jaata|padhta|sochta|aata|khelta|rehta)\s+(?:hu|hoon|hun)\b', t) or \
       re.search(r'\b(?:i am|i\'m)\s+(?:a\s+)?(?:guy|boy|man|male)\b', t) or \
       re.search(r'\b(?:ladka|boy)\s+(?:hu|hoon|hun)\b', t):
        return 'male'
    # 1st person feminine verb endings: kr rhi hu, kar rahi hoon, karti hu, etc.
    if re.search(r'\b(?:kr|kar|ja|dekh|baithi|padh|soch|aa|khel|khaa|ruk|le|de)\s*(?:rahi|rhi)\s+(?:hu|hoon|hun)\b', t) or \
       re.search(r'\b(?:karti|jaati|padhti|sochti|aati|khelti|rehti)\s+(?:hu|hoon|hun)\b', t) or \
       re.search(r'\b(?:i am|i\'m)\s+(?:a\s+)?(?:girl|woman|female)\b', t) or \
       re.search(r'\b(?:ladki|girl)\s+(?:hu|hoon|hun)\b', t):
        return 'female'
    return None


# ---------------------------------------------------------------------------
# Regex Pattern Matchers for High-Precision Memory Extraction
# (Supports both English and Hinglish expressions)
# ---------------------------------------------------------------------------

PATTERNS = {
    "name": [
        (re.compile(r"\bmy name is\s+([A-Za-z]+)", re.I), "fact:name", "User's name is {0}"),
        (re.compile(r"\bcall me\s+([A-Za-z]+)", re.I), "rel:nickname", "User prefers to be called {0}"),
        (re.compile(r"\bmera naam\s+([A-Za-z]+)\s+hai", re.I), "fact:name", "User's name is {0}"),
    ],
    "occupation": [
        (re.compile(r"\bi (?:work as|am an?)\s+(?:a |an )?([A-Za-z\s]+?)(?:\.|$|,|\b(?:at|for|in)\b)", re.I), "fact:occupation", "User works as {0}"),
        (re.compile(r"\bmy (?:job|profession|work|career) is\s+([A-Za-z\s]+?)(?:\.|$|,)", re.I), "fact:occupation", "User's profession is {0}"),
        (re.compile(r"\bmain\s+([A-Za-z\s]+?)\s+(?:ka kaam karta hu|hu|me job karta hu)", re.I), "fact:occupation", "User works as {0}"),
    ],
    "location": [
        (re.compile(r"\bi (?:live in|stay in|am from|grew up in)\s+([A-Za-z\s]+?)(?:\.|$|,)", re.I), "fact:location", "User lives in/from {0}"),
        (re.compile(r"\b(?:mai|main)\s+(?:toh\s+|to\s+|bhi\s+)?([A-Za-z\s]+?)\s+(?:me\s+(?:rehta|rehti)\s+(?:hu|hoon|hun)|se\s+(?:hu|hoon|hun))\b", re.I), "fact:location", "User lives in/from {0}"),
    ],
    "age": [
        (re.compile(r"\bi(?:'m| am)\s+(\d{1,2})\s*(?:years old|yrs old|yo)?", re.I), "fact:age", "User is {0} years old"),
        (re.compile(r"\bmeri age\s+(\d{1,2})\s*hai", re.I), "fact:age", "User is {0} years old"),
    ],
    "preferences": [
        (re.compile(r"\bi (?:really love|love|adore)\s+([A-Za-z\s]+?)(?:\.|$|,|\bbecause\b)", re.I), "pref:love", "User loves {0}"),
        (re.compile(r"\bi (?:really hate|hate|dislike|can't stand)\s+([A-Za-z\s]+?)(?:\.|$|,)", re.I), "pref:hate", "User dislikes {0}"),
        (re.compile(r"\bmy favorite\s+([A-Za-z\s]+?)\s+is\s+([A-Za-z0-9\s]+?)(?:\.|$|,)", re.I), "pref:favorite", "User's favorite {0} is {1}"),
        (re.compile(r"\bmujhe\s+([A-Za-z\s]+?)\s+(?:bohot pasand hai|pasand hai)", re.I), "pref:like", "User likes {0}"),
        (re.compile(r"\bmera favorite\s+([A-Za-z\s]+?)\s+([A-Za-z0-9\s]+?)\s+hai", re.I), "pref:favorite", "User's favorite {0} is {1}"),
        (re.compile(r"\b(?:mujhe\s+)?(short|chote|brief)\s+replies\s+(?:pasand|chahiye|do)", re.I), "pref:reply_length", "User prefers concise, short replies"),
        (re.compile(r"\b(?:please\s+)?(?:keep\s+replies\s+short|give\s+short\s+replies)", re.I), "pref:reply_length", "User prefers concise, short replies"),
        (re.compile(r"\b(?:talk in|speak in|baat karo)\s+(hinglish|hindi|english)", re.I), "pref:language", "User prefers to chat in {0}"),
    ],
    "episodic": [
        (re.compile(r"\bi have an? (interview|exam|presentation|meeting|date|flight)\s*(tomorrow|next week|today|on [A-Za-z]+)?", re.I), "event:upcoming", "User has an upcoming {0} {1}"),
        (re.compile(r"\b(kal|parso)\s+mera\s+([A-Za-z]+)\s+hai", re.I), "event:upcoming", "User has an event ({1}) coming up {0}"),
        (re.compile(r"\bi(?:'m| am) traveling to\s+([A-Za-z\s]+?)\s+(?:next week|tomorrow|soon)", re.I), "event:travel", "User is traveling to {0}"),
        (re.compile(r"\bmain\s+([A-Za-z\s]+?)\s+(?:ja raha hu|ja rahi hu)", re.I), "event:travel", "User is visiting {0}"),
    ],
    "emotion": [
        (re.compile(r"\bi(?:'m| am| feel)\s+(?:so |very |really )?(nervous|anxious|stressed|sad|depressed|excited|happy|thrilled|exhausted|tired)", re.I), "cue:emotion", "{0}"),
        (re.compile(r"\b(?:bohot |thoda )?(nervous|stressed|khush|sad|tension|anxious|tired)\s+(?:hu|feel ho raha hai|ho raha hai)", re.I), "cue:emotion", "{0}"),
    ],
}


class MemoryCandidate:
    """Represents an extracted prospective memory item before persistence."""
    def __init__(
        self,
        memory_type: str,
        memory_text: str,
        normalized_key: str,
        confidence: float = 0.9,
        importance: float = 0.6,
        source_type: str = "explicit",
        profile_id: Optional[str] = None,
        expires_at: Optional[datetime] = None,
    ):
        self.memory_type = memory_type
        self.memory_text = memory_text.strip()
        self.normalized_key = normalized_key
        self.confidence = confidence
        self.importance = importance
        self.source_type = source_type
        self.profile_id = profile_id
        self.expires_at = expires_at

    def __repr__(self):
        return f"<MemoryCandidate {self.normalized_key}: {self.memory_text[:30]}>"


def extract_candidate_memories(
    user_message: str,
    profile_id: str,
) -> Tuple[List[MemoryCandidate], Optional[str], Optional[str]]:
    """
    Parse user message into structured candidates, detected emotional cue, and active topic.
    Returns: (candidates_list, detected_emotion, detected_topic)
    """
    candidates: List[MemoryCandidate] = []
    detected_emotion: Optional[str] = None
    detected_topic: Optional[str] = None
    clean_msg = user_message.strip()

    if len(clean_msg) < 4:
        return candidates, None, None

    # 1. Name & Identity
    for pattern, key, template in PATTERNS["name"]:
        m = pattern.search(clean_msg)
        if m:
            val = m.group(1).strip().capitalize()
            if len(val) >= 2 and val.lower() not in ["not", "no", "just", "very", "a"]:
                mem_type = "relationship" if "nickname" in key else "profile_fact"
                p_id = profile_id if mem_type == "relationship" else None
                candidates.append(MemoryCandidate(
                    memory_type=mem_type,
                    memory_text=template.format(val),
                    normalized_key=f"{key}:{val.lower()}",
                    confidence=0.98,
                    importance=0.9,
                    profile_id=p_id,
                ))

    # 2. Occupation
    for pattern, key, template in PATTERNS["occupation"]:
        m = pattern.search(clean_msg)
        if m:
            val = m.group(1).strip()
            if len(val) >= 3 and val.lower() not in ["good", "bad", "okay", "fine"]:
                candidates.append(MemoryCandidate(
                    memory_type="profile_fact",
                    memory_text=template.format(val),
                    normalized_key="fact:occupation",
                    confidence=0.92,
                    importance=0.75,
                ))
                detected_topic = f"job: {val}"

    # 3. Location
    for pattern, key, template in PATTERNS["location"]:
        m = pattern.search(clean_msg)
        if m:
            val = m.group(1).strip()
            # Strip conversational particles: to, toh, bhi
            val = re.sub(r"^(?:to|toh|bhi)\s+", "", val, flags=re.I).strip().title()
            val_lower = val.lower()
            invalid_locs = {
                "home", "work", "here", "there", "kis", "kis city", "kahan", "kaha",
                "kon", "kaun", "which", "what", "where", "city", "shehar", "kisi", "bhi"
            }
            if len(val) >= 3 and val_lower not in invalid_locs and not any(q in val_lower for q in ["kis ", "which "]):
                candidates.append(MemoryCandidate(
                    memory_type="profile_fact",
                    memory_text=template.format(val),
                    normalized_key=f"fact:location:{val_lower}",
                    confidence=0.95,
                    importance=0.7,
                ))

    # 4. Age
    for pattern, key, template in PATTERNS["age"]:
        m = pattern.search(clean_msg)
        if m:
            age_val = m.group(1).strip()
            if 16 <= int(age_val) <= 99:
                candidates.append(MemoryCandidate(
                    memory_type="profile_fact",
                    memory_text=template.format(age_val),
                    normalized_key="fact:age",
                    confidence=0.95,
                    importance=0.7,
                )) 

    # 5. Preferences
    for pattern, key, template in PATTERNS["preferences"]:
        m = pattern.search(clean_msg)
        if m:
            groups = [g.strip() for g in m.groups() if g]
            if groups:
                pref_text = template.format(*groups)
                norm_sub = re.sub(r"[^a-z0-9]", "_", groups[-1].lower())[:30]
                candidates.append(MemoryCandidate(
                    memory_type="preference",
                    memory_text=pref_text,
                    normalized_key=f"{key}:{norm_sub}",
                    confidence=0.88,
                    importance=0.65,
                ))
                detected_topic = f"preference: {groups[0]}"

    # 6. Episodic Events (e.g. Interviews, exams, travel)
    for pattern, key, template in PATTERNS["episodic"]:
        m = pattern.search(clean_msg)
        if m:
            groups = [g.strip() for g in m.groups() if g]
            if groups:
                event_name = groups[0]
                timing = groups[1] if len(groups) > 1 else "soon"
                event_text = f"User has an upcoming {event_name} ({timing})"
                # Default expiration: 3 days after event for near-term follow-up
                exp_date = datetime.utcnow() + timedelta(days=3)
                norm_key = f"event:{re.sub(r'[^a-z0-9]', '_', event_name.lower())[:25]}"
                candidates.append(MemoryCandidate(
                    memory_type="episodic",
                    memory_text=event_text,
                    normalized_key=norm_key,
                    confidence=0.90,
                    importance=0.85,
                    expires_at=exp_date,
                    profile_id=profile_id,
                ))
                detected_topic = f"upcoming {event_name}"

    # 7. Emotional Cues
    for pattern, key, template in PATTERNS["emotion"]:
        m = pattern.search(clean_msg)
        if m:
            emo = m.group(1).strip().lower()
            detected_emotion = emo
            break

    # 8. Pronoun Preference
    p_pref = detect_pronoun_preference(clean_msg)
    if p_pref:
        candidates.append(MemoryCandidate(
            memory_type="preference",
            memory_text=f"User prefers companion to address them with '{p_pref}'",
            normalized_key="pref:pronoun",
            confidence=0.98,
            importance=0.95,
        ))

    # 9. User Gender Cue
    g_cue = detect_user_gender_cue(clean_msg)
    if g_cue:
        candidates.append(MemoryCandidate(
            memory_type="profile_fact",
            memory_text=f"User is {g_cue}",
            normalized_key="fact:gender",
            confidence=0.95,
            importance=0.85,
        ))

    # 10. Contextual Topic Heuristics (e.g. office workload, weekend plans)
    if not detected_topic:
        lower_msg = clean_msg.lower()
        if any(w in lower_msg for w in ["office", "kaam", "work", "job", "workload", "boss"]):
            detected_topic = "office work and workload"
        elif any(w in lower_msg for w in ["weekend", "sunday", "saturday", "rest", "chhutti", "holiday"]):
            detected_topic = "weekend plans and rest"

    return candidates, detected_emotion, detected_topic


def process_and_persist_memories(
    db: Session,
    user_id: str,
    profile_id: str,
    user_message: str,
    source_message_id: Optional[int] = None,
) -> List[UserMemory]:
    """
    Orchestrate full extraction, deduplication, SQLite persistence,
    ChromaDB vector sync, and conversation state update.
    """
    persisted: List[UserMemory] = []

    try:
        candidates, emotion, topic = extract_candidate_memories(user_message, profile_id)

        # 1. Update Ephemeral Conversation State
        from app.chat.prompt_builder import is_user_correction_message, is_user_contradiction_message
        recent_ctx_updates = {}
        if is_user_contradiction_message(user_message):
            recent_ctx_updates["active_style_feedback"] = "User identified a factual contradiction. Stick strictly to verified persona facts without inventing institutions."
        elif is_user_correction_message(user_message):
            recent_ctx_updates["active_style_feedback"] = "User requested direct and concise responses without unsolicited details or commentary."

        pronoun = detect_pronoun_preference(user_message)
        if pronoun:
            recent_ctx_updates["preferred_pronoun"] = pronoun

        gender = detect_user_gender_cue(user_message)
        if gender:
            recent_ctx_updates["user_gender"] = gender

        if emotion or topic or recent_ctx_updates:
            update_conversation_state(
                db=db,
                user_id=user_id,
                profile_id=profile_id,
                current_topic=topic if topic else None,
                user_stated_emotion=emotion if emotion else None,
                inferred_emotional_cue=emotion if emotion else None,
                open_threads=[topic] if topic else None,
                recent_context=recent_ctx_updates if recent_ctx_updates else None,
            )

        # 2. Process and Persist Each Candidate Memory
        for cand in candidates:
            # Check for contradiction / existing normalized_key
            saved = save_user_memory(
                db=db,
                user_id=user_id,
                memory_type=cand.memory_type,
                memory_text=cand.memory_text,
                profile_id=cand.profile_id,
                normalized_key=cand.normalized_key,
                importance=cand.importance,
                confidence=cand.confidence,
                source_type=cand.source_type,
                expires_at=cand.expires_at,
                source_message_id=source_message_id,
            )
            persisted.append(saved)

            # 3. Synchronously Sync to ChromaDB for RAG semantic search
            try:
                store_memory_fact(
                    profile_id=cand.profile_id or profile_id,
                    user_id=user_id,
                    fact=cand.memory_text,
                    fact_id=saved.id,
                )
            except Exception as v_exc:
                logger.warning("Vector store sync for memory %s warning: %s", saved.id, v_exc)

        if persisted:
            logger.info("Extracted and persisted %d long-term memories for user %s", len(persisted), user_id[:8])

    except Exception as exc:
        # Non-fatal: memory extraction failure must NEVER cause chat failure
        logger.error("Memory extraction error for user %s: %s", user_id, exc, exc_info=True)

    return persisted
