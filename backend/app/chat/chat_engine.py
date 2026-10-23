"""
chat/chat_engine.py
--------------------
The central orchestrator of the AI chat pipeline.

Flow:
  1. Load persona configuration.
  2. Retrieve RAG grounding facts & scored long-term user memories.
  3. Fetch rolling summary & active conversation state.
  4. Fetch recent conversation history turns.
  5. Assemble modular prompt with human texting guidelines & language matching.
  6. Generate AI response via resilient LLM router (Groq -> Claude fallback).
  7. Validate and sanitize generated response (no metadata leaks, no consecutive repetitions).
  8. Persist conversation turn in SQLite messages table.
  9. Run asynchronous/safe long-term memory extraction and state update.
"""

import logging
import json
import re
from datetime import datetime
from typing import Any, Optional, Tuple
from sqlalchemy.orm import Session

from app.personas.persona_loader import load_persona
from app.rag.retriever import retrieve_context
from app.memory.conversation_memory import get_conversation_history, persist_exchange
from app.memory.memory_store import (
    ensure_user, ensure_profile, get_latest_summary, get_or_create_conversation_state,
    update_conversation_state
)
from app.memory.memory_extractor import (
    process_and_persist_memories, detect_pronoun_preference, detect_user_gender_cue
)
from app.config.settings import settings
from app.utils.time_context import get_current_context
from app.chat.context_engine import build_live_context
from app.chat.topic_picker import get_topic_suggestion, update_topics_state
from app.llm.openai_client import record_regeneration
from app.chat.response_validator import (
    validate_and_clean_response,
    arrange_and_format_response,
    clean_banned_metaphors,
    count_banned_metaphors,
    is_user_in_distress,
    sanitize_emojis,
)
from app.chat.own_life_engine import should_inject_own_life, get_own_life_seed
from app.chat.prompt_builder import build_final_prompt, build_direct_prompt, is_low_energy_response
from app.llm.llm_router import route_and_generate
from app.utils.error_handler import PersonaNotFoundError, LLMClientError

logger = logging.getLogger(__name__)


async def process_chat_message(
    db: Session,
    profile_id: str,
    user_id: str,
    user_message: str,
    background_tasks: Any = None,
) -> str:
    """
    Full pipeline: receive a user message and return the AI persona's reply.
    """
    logger.info("Chat request — profile: %s | user: %s", profile_id, user_id[:8])

    # 1. Load persona config
    persona = load_persona(profile_id)
    persona_name = persona.get("name", "Companion")

    # 2. Ensure DB records exist for this user+profile
    ensure_user(db, user_id)
    ensure_profile(db, profile_id, persona_name, persona.get("short_bio", ""))

    # 3. Retrieve RAG grounding context & scored long-term user memories
    rag_context = retrieve_context(
        profile_id=profile_id,
        user_message=user_message,
        user_id=user_id,
        db=db,
    )

    # 4. Fetch rolling conversation summary & active conversation state
    summary_record = get_latest_summary(db, user_id, profile_id)
    rolling_summary = summary_record.summary_text if summary_record else None

    # Authoritative runtime date/time context
    time_context = get_current_context()

    state_record = get_or_create_conversation_state(db, user_id, profile_id)
    recent_ctx = {}
    try:
        recent_ctx = json.loads(state_record.recent_context_json or "{}")
    except Exception:
        recent_ctx = {}

    # Extract pronoun preference (default "tum") & user grammatical gender
    preferred_pronoun = recent_ctx.get("preferred_pronoun", "tum")
    user_gender = recent_ctx.get("user_gender")

    # Dynamic detection from current message
    msg_pronoun = detect_pronoun_preference(user_message)
    if msg_pronoun:
        preferred_pronoun = msg_pronoun
        recent_ctx["preferred_pronoun"] = msg_pronoun

    msg_gender = detect_user_gender_cue(user_message)
    if msg_gender:
        user_gender = msg_gender
        recent_ctx["user_gender"] = msg_gender

    # 5. Fetch recent conversation history turns
    chat_history = get_conversation_history(db, user_id, profile_id)

    # If gender wasn't in state or current message, inspect recent user turns
    if not user_gender and chat_history:
        for turn in reversed(chat_history):
            if turn.get("role") == "user":
                hist_gender = detect_user_gender_cue(turn.get("content", ""))
                if hist_gender:
                    user_gender = hist_gender
                    recent_ctx["user_gender"] = hist_gender
                    break

    # Session staleness check: If last session was > 4 hours ago, do not blindly carry forward old topics
    is_stale_session = False
    if state_record.updated_at:
        delta_sec = (datetime.utcnow() - state_record.updated_at).total_seconds()
        if delta_sec > 14400:
            is_stale_session = True

    # Compute message count & relationship stage
    total_messages = len(chat_history)
    rel_stage = "new" if total_messages <= 20 else ("getting_close" if total_messages <= 60 else "comfortable")

    topics_discussed = recent_ctx.get("topics_discussed", {})
    last_topic_at = recent_ctx.get("last_topic_suggestion_at", 0)
    persona_mood = recent_ctx.get("persona_mood", "chill")

    state_dict = {
        "current_topic": None if is_stale_session else state_record.current_topic,
        "user_stated_emotion": state_record.user_stated_emotion,
        "open_threads": [] if is_stale_session else json.loads(state_record.open_threads_json or "[]"),
        "active_style_feedback": recent_ctx.get("active_style_feedback"),
        "preferred_pronoun": preferred_pronoun,
        "user_gender": user_gender,
        "topics_discussed": topics_discussed,
        "last_topic_suggestion_at": last_topic_at,
        "persona_mood": persona_mood,
        "relationship_stage": rel_stage,
    }

    # 6. Topic engine & Live context
    topic_suggestion = get_topic_suggestion(
        user_message=user_message,
        conversation_state=state_dict,
        message_count=total_messages,
        persona_name=persona_name,
    )
    if topic_suggestion:
        state_dict = update_topics_state(
            state_dict,
            topic_suggestion.get("topic_id"),
            total_messages,
        )

    live_ctx = build_live_context()

    # Care Mode state tracking (2-3 turn persistence upon distress/sickness/sadness)
    user_distress = is_user_in_distress(user_message)
    current_care_turns = recent_ctx.get("care_mode_turns_remaining", 0)
    if user_distress:
        care_mode_turns_remaining = 3
        is_care_mode = True
    elif current_care_turns > 0:
        care_mode_turns_remaining = current_care_turns - 1
        is_care_mode = True
    else:
        care_mode_turns_remaining = 0
        is_care_mode = False

    # Short reply tease counter (never comment on short reply more than once per conversation)
    short_reply_tease_count = recent_ctx.get("short_reply_tease_count", 0)

    # Humor budget: at most 1 in 4 replies can contain a joke/metaphor/wordplay
    turns_since_last_joke = recent_ctx.get("turns_since_last_joke", 3)
    allow_humor = (turns_since_last_joke >= 3) and (not is_care_mode)

    # Emoji rule: at most 1 in 3 replies, never same emoji twice, never laughing emoji in care mode
    turns_since_last_emoji = recent_ctx.get("turns_since_last_emoji", 2)
    last_bot_emoji = recent_ctx.get("last_bot_emoji", "")
    allow_emoji = (turns_since_last_emoji >= 2) and (not is_care_mode)

    # Own life engine: share real daily detail in first 4-5 msgs, then every ~10 msgs
    last_self_share_at = recent_ctx.get("last_self_share_at", 0)
    own_life_seed = None
    if not is_care_mode and should_inject_own_life(total_messages, last_self_share_at):
        try:
            from app.utils.time_context import get_runtime_time_context
            rt_ctx = get_runtime_time_context()
            time_str = rt_ctx.get("formatted_time", "11:00 AM")
            day_str = rt_ctx.get("day_name", "Friday")
            own_life_seed = get_own_life_seed(persona_name, current_time=time_str, day_of_week=day_str)
        except Exception as seed_err:
            logger.debug("Failed to get own life seed: %s", seed_err)

    # Structure variety tracking: track last 5 reply shapes and avoid repeating two-line splits
    recent_reply_shapes = recent_ctx.get("recent_reply_shapes", [])
    last_shape = recent_reply_shapes[-1] if recent_reply_shapes else None
    force_single_line = (last_shape == "two_line")

    # Build prompt (Direct rule-free flow or modular prompt)
    if getattr(settings, "use_direct_prompt_flow", True):
        logger.debug("Using direct rule-free prompt flow for profile %s", profile_id)
        system_prompt, trimmed_history = build_direct_prompt(
            persona=persona,
            chat_history=chat_history,
            user_message=user_message,
            rag_context=rag_context,
            rolling_summary=rolling_summary,
            conversation_state=state_dict,
            time_context=time_context,
            topic_suggestion=topic_suggestion,
            live_context=live_ctx,
            is_care_mode=is_care_mode,
            short_reply_tease_count=short_reply_tease_count,
            last_reply_shape=last_shape,
            allow_humor=allow_humor,
            own_life_seed=own_life_seed,
        )
    else:
        system_prompt, trimmed_history = build_final_prompt(
            persona=persona,
            rag_context=rag_context,
            chat_history=chat_history,
            user_message=user_message,
            rolling_summary=rolling_summary,
            conversation_state=state_dict,
            time_context=time_context,
        )

    # 7. Generate AI response (OpenAI direct / router)
    raw_reply = route_and_generate(
        db=db,
        system_prompt=system_prompt,
        chat_history=trimmed_history,
        user_message=user_message,
        user_id=user_id,
    )

    # 7b. Post-processor single regeneration trigger
    # Trigger on:
    #   - reply length > max(user_words * 2, 20)
    #   - 2+ praise/agreement phrases
    #   - 2+ banned metaphor punchlines ("trailer mode", "negotiation", "budget", "vibe hijack", "araam mode", "scene on hai")
    #   - care mode violation (laughing emoji or banned metaphor during illness/sadness)
    user_words = len(user_message.strip().split())
    max_allowed_words = max(user_words * 2, 20)
    reply_words = len(raw_reply.strip().split())

    _PRAISE_PATTERNS = [
        r"\bsolid\s+choice\b",
        r"\bgreat\s+choice\b",
        r"\bbadhiya\s+choice\b",
        r"\bsahi\s+choice\b",
        r"\bbilkul\s+sahi\b",
        r"\bsahi\s+kaha\b",
        r"\bkya\s+baat\s+hai\b",
        r"\bkoi\s+comparison\s+nahi\b",
    ]
    praise_matches = sum(len(re.findall(pat, raw_reply, re.IGNORECASE)) for pat in _PRAISE_PATTERNS)
    banned_metaphors_count = count_banned_metaphors(raw_reply)
    has_laughing_in_care = bool(re.search(r"[😂😅🤣😜🤪😹😆]", raw_reply)) if is_care_mode else False

    should_regenerate = (
        (reply_words > max_allowed_words)
        or (praise_matches >= 2)
        or (banned_metaphors_count >= 2)
        or (is_care_mode and (has_laughing_in_care or banned_metaphors_count >= 1))
    )

    if should_regenerate:
        logger.info(
            "Regeneration triggered for user %s: words=%d/%d, praise=%d, metaphors=%d, care_laugh=%s. Retrying once.",
            user_id[:8], reply_words, max_allowed_words, praise_matches, banned_metaphors_count, has_laughing_in_care,
        )
        record_regeneration(True)
        if is_care_mode:
            regeneration_nudge = (
                "\n\n[REGENERATION NUDGE — CARE MODE]: The user is unwell, stressed, or sad. "
                "Your previous reply had laughing emojis, banned metaphors, or jokes. "
                "Regenerate: Show genuine, simple care (e.g. 'arey yaar, bukhar hai kya? dawai li?'). "
                "ZERO jokes, ZERO metaphors, ZERO laughing emojis. Keep it under 15 words."
            )
        elif banned_metaphors_count >= 2:
            regeneration_nudge = (
                "\n\n[REGENERATION NUDGE — BANNED METAPHORS]: Your previous reply used try-hard metaphor punchlines "
                "(e.g. trailer mode, negotiation, budget, vibe hijack, araam mode). "
                "Regenerate: Keep it completely plain and natural like real casual WhatsApp texting "
                "(e.g. 'haha sach me?', 'kya hua', 'arre yaar', 'phir?'). NO metaphor punchlines."
            )
        else:
            regeneration_nudge = (
                "\n\n[REGENERATION NUDGE]: Your previous attempt was too long or contained generic praise phrases. "
                "Please regenerate: KEEP IT UNDER 15 WORDS. DO NOT praise or flatter. "
                "Text like a real person chatting on WhatsApp — tease them, disagree playfully, or react with personality."
            )
        try:
            raw_reply = route_and_generate(
                db=db,
                system_prompt=system_prompt + regeneration_nudge,
                chat_history=trimmed_history,
                user_message=user_message,
                user_id=user_id,
            )
        except Exception as regen_err:
            logger.warning("Regeneration attempt failed: %s; keeping initial reply", regen_err)
    else:
        record_regeneration(False)

    # 8. Deterministic response arrangement & formatting according to user
    ai_reply = arrange_and_format_response(
        raw_response=raw_reply,
        persona_name=persona_name,
        recent_history=trimmed_history,
        user_gender=user_gender,
        preferred_pronoun=preferred_pronoun,
        user_message=user_message,
        allow_emoji=allow_emoji,
        last_bot_emoji=last_bot_emoji,
        is_care_mode=is_care_mode,
        force_single_line=force_single_line,
    )

    # 9. Persist conversation turn in SQLite
    user_msg_record, asst_msg_record = persist_exchange(
        db=db,
        user_id=user_id,
        profile_id=profile_id,
        user_message=user_message,
        ai_reply=ai_reply,
    )

    # Update state trackers for next turns
    if is_low_energy_response(user_message):
        short_reply_tease_count += 1

    had_joke = bool(re.search(r"\b(haha|hehe|lol|joke|kya scene)\b", ai_reply, re.I))
    if had_joke and allow_humor:
        turns_since_last_joke = 0
    else:
        turns_since_last_joke += 1

    found_emojis = re.findall(r"[\U00010000-\U0010ffff]", ai_reply)
    if found_emojis:
        turns_since_last_emoji = 0
        last_bot_emoji = found_emojis[0]
    else:
        turns_since_last_emoji += 1

    if own_life_seed:
        last_self_share_at = total_messages

    current_shape = "two_line" if "\n" in ai_reply else ("short_single" if len(ai_reply.split()) <= 8 else "regular_single")
    recent_reply_shapes = (recent_reply_shapes + [current_shape])[-5:]

    # Keep conversation-scoped cues & topic state synchronously available for next turn
    state_updates = {
        "last_user_message": user_message.strip()[:240],
        "last_assistant_message": ai_reply[:240],
        "topics_discussed": state_dict.get("topics_discussed", {}),
        "last_topic_suggestion_at": state_dict.get("last_topic_suggestion_at", 0),
        "persona_mood": state_dict.get("persona_mood", "chill"),
        "relationship_stage": state_dict.get("relationship_stage", "new"),
        "care_mode_turns_remaining": care_mode_turns_remaining,
        "short_reply_tease_count": short_reply_tease_count,
        "turns_since_last_joke": turns_since_last_joke,
        "turns_since_last_emoji": turns_since_last_emoji,
        "last_bot_emoji": last_bot_emoji,
        "last_self_share_at": last_self_share_at,
        "recent_reply_shapes": recent_reply_shapes,
    }
    if state_dict.get("current_topic") or user_message.strip():
        update_conversation_state(
            db=db,
            user_id=user_id,
            profile_id=profile_id,
            current_topic=state_dict.get("current_topic") or user_message.strip()[:120],
            conversation_tone=state_dict.get("conversation_tone") or "casual",
            recent_context=state_updates,
        )

    # 10. Extract, deduplicate, and persist long-term candidate memories
    # Use non-blocking background task so HTTP response returns immediately
    source_msg_id = user_msg_record.id if user_msg_record else None
    if background_tasks:
        def _bg_worker(u_id: str, p_id: str, msg: str, s_id: Any):
            from app.api.dependencies import SessionLocal
            bg_db = SessionLocal()
            try:
                process_and_persist_memories(
                    db=bg_db,
                    user_id=u_id,
                    profile_id=p_id,
                    user_message=msg,
                    source_message_id=s_id,
                )
            except Exception as bg_exc:
                logger.warning("Background memory extraction error: %s", bg_exc)
            finally:
                bg_db.close()

        background_tasks.add_task(_bg_worker, user_id, profile_id, user_message, source_msg_id)
        logger.debug("Scheduled background memory extraction for user %s", user_id[:8])
    else:
        # Fallback to in-process execution (e.g. in test suites)
        process_and_persist_memories(
            db=db,
            user_id=user_id,
            profile_id=profile_id,
            user_message=user_message,
            source_message_id=source_msg_id,
        )

    logger.info("Reply generated & persisted (%d chars) for profile %s", len(ai_reply), profile_id)
    return ai_reply


_CONVERSATION_END_PATTERNS = re.compile(
    r"\b("
    r"bye|byee+|good\s*night|gn|alvida|chalo\s+bye|so\s+raha\s+hu|so\s+rahi\s+hu|"
    r"so\s+jao|baad\s+me\s+baat|ttyl|cya|see\s+you|talk\s+later|tata"
    r")\b",
    re.IGNORECASE,
)


async def process_nudge_message(
    db: Session,
    profile_id: str,
    user_id: str,
    min_idle_seconds: int = 180,
) -> Tuple[Optional[str], str]:
    """
    Check if user went idle without ending conversation (3-4 minutes)
    and generate exactly ONE proactive re-engagement check-in from the companion.

    Returns (reply_text, reason)
    """
    from app.memory.memory_store import get_full_history
    from app.models.db_models import Message

    # 1. Load persona
    persona = load_persona(profile_id)
    persona_name = persona.get("name", "Companion")

    # 2. Get full conversation history
    messages = get_full_history(db, user_id, profile_id)
    if not messages:
        return None, "no_messages"

    user_msgs = [m for m in messages if m.role == "user"]
    if not user_msgs:
        return None, "no_user_messages"

    last_user_msg = user_msgs[-1]
    last_overall_msg = messages[-1]

    # 3. Check if user ended the conversation gracefully
    if _CONVERSATION_END_PATTERNS.search(last_user_msg.content or ""):
        logger.info("User ended conversation gracefully; skipping nudge.")
        return None, "user_ended_conversation"

    # 4. Check if we already nudged for this user message
    state_record = get_or_create_conversation_state(db, user_id, profile_id)
    recent_ctx = {}
    try:
        recent_ctx = json.loads(state_record.recent_context_json or "{}")
    except Exception:
        recent_ctx = {}

    if recent_ctx.get("nudged_for_user_msg_id") == str(last_user_msg.id):
        logger.info("Already nudged for user message %s; skipping.", last_user_msg.id)
        return None, "already_nudged"

    # 5. Check idle threshold
    now_utc = datetime.utcnow()
    msg_time = last_overall_msg.created_at
    if msg_time and msg_time.tzinfo is not None:
        msg_time = msg_time.replace(tzinfo=None)
    elapsed = (now_utc - msg_time).total_seconds() if msg_time else 0
    if elapsed < min_idle_seconds:
        return None, "idle_threshold_not_reached"

    # 6. Retrieve RAG grounding, summary & context
    rag_context = retrieve_context(profile_id=profile_id, user_message="kahan ho", user_id=user_id, db=db)
    summary_record = get_latest_summary(db, user_id, profile_id)
    rolling_summary = summary_record.summary_text if summary_record else None
    time_context = get_current_context()

    preferred_pronoun = recent_ctx.get("preferred_pronoun", "tum")
    user_gender = recent_ctx.get("user_gender")

    chat_history = get_conversation_history(db, user_id, profile_id)

    state_dict = {
        "current_topic": state_record.current_topic,
        "user_stated_emotion": state_record.user_stated_emotion,
        "open_threads": json.loads(state_record.open_threads_json or "[]"),
        "preferred_pronoun": preferred_pronoun,
        "user_gender": user_gender,
    }

    # 7. Build prompt with INACTIVITY NUDGE directive
    system_prompt, trimmed_history = build_direct_prompt(
        persona=persona,
        chat_history=chat_history,
        user_message="[User went silent 3-4 minutes ago]",
        rag_context=rag_context,
        rolling_summary=rolling_summary,
        conversation_state=state_dict,
        time_context=time_context,
    )

    nudge_directive = (
        "CONVERSATION INACTIVITY RE-ENGAGEMENT NUDGE (CRITICAL): The user was chatting with you, "
        "but has gone silent for 3-4 minutes without saying bye. "
        "Send ONE short, casual, witty check-in (5-12 words max, 1 single short line). "
        "Ask where they disappeared or playfully tease them about where they went or about the last topic. "
        "Examples: "
        "- 'kahan gayab ho gaye achanak se? haha 😂' "
        "- 'arey kahan chale gaye? itna kya sochne lage haha' "
        "- 'gayab ho gaye kya? 😭 main yahan wait kar rahi hu' "
        "- (If previous topic was horror): 'horror ka naam sunte hi bhaag gaye kya haha 😂' "
        "Rules: "
        "- Do NOT be needy, desperate, or formal. "
        "- Do NOT ask multiple questions. "
        "- Use the EXACT same language/script (Hinglish/Hindi/English) as previous messages. "
        "- Keep it to 1 single short casual line."
    )
    system_prompt = system_prompt + "\n\n" + nudge_directive

    # 8. Generate response via router
    raw_reply = route_and_generate(
        db=db,
        system_prompt=system_prompt,
        chat_history=trimmed_history,
        user_message="[User has been silent for 3-4 minutes]",
    )

    # 9. Clean & format
    ai_reply = arrange_and_format_response(
        raw_response=raw_reply,
        persona_name=persona_name,
        recent_history=trimmed_history,
        user_gender=user_gender,
        preferred_pronoun=preferred_pronoun,
        user_message="[User silent]",
    )

    # 10. Persist single assistant message
    asst_msg = Message(
        user_id=user_id,
        profile_id=profile_id,
        role="assistant",
        content=ai_reply,
        created_at=datetime.utcnow(),
    )
    db.add(asst_msg)
    db.commit()
    db.refresh(asst_msg)

    # Update conversation state to remember nudge was sent for this turn
    recent_ctx["nudged_for_user_msg_id"] = str(last_user_msg.id)
    recent_ctx["last_assistant_message"] = ai_reply
    update_conversation_state(
        db=db,
        user_id=user_id,
        profile_id=profile_id,
        current_topic=state_record.current_topic or "idle_check",
        conversation_tone="casual",
        recent_context=recent_ctx,
    )

    logger.info("Nudge generated & persisted for profile %s: '%s'", profile_id, ai_reply[:40])
    return ai_reply, "success"

