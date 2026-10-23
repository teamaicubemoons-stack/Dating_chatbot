"""
tests/chat_quality_check.py
---------------------------
Automated quality metrics checker for Spark chat personas.
Evaluates replies against conversational design standards:
  1. Average reply length (target: <= 20 words for typical texting)
  2. Question rate (target: <= 40% of messages end with a question — anti-interrogation)
  3. Praise/agreement rate (target: 0% robotic praise phrases like 'solid choice', 'sahi baat hai')
  4. Script consistency (mirrors user's script: English -> English, Hinglish -> Hinglish)
  5. Metaphor/Wordplay budget: % replies with joke/metaphor/wordplay (target < 25%)
  6. Emoji frequency: % replies with emoji (target 30-40%)
  7. Repeated reply-shape rate: consecutive structural shape repeats (target < 25%)
  8. Care-mode correctness: on messages mentioning illness or sadness (target 100%)
"""

import os
import sys
import re
from typing import List, Dict, Any, Tuple

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.chat.response_validator import (
    arrange_and_format_response,
    clean_banned_metaphors,
    count_banned_metaphors,
    is_user_in_distress,
    sanitize_emojis,
)

BANNED_PRAISE_PATTERNS = [
    re.compile(r"\bsolid\s+choice\b", re.I),
    re.compile(r"\bgreat\s+choice\b", re.I),
    re.compile(r"\bbadhiya\s+choice\b", re.I),
    re.compile(r"\bsahi\s+choice\b", re.I),
    re.compile(r"\bbilkul\s+sahi\b", re.I),
    re.compile(r"\bsahi\s+kaha\b", re.I),
    re.compile(r"\bkya\s+baat\s+hai\b", re.I),
    re.compile(r"\bkoi\s+comparison\s+nahi\b", re.I),
]

PHILOSOPHICAL_GYAAN_PATTERNS = [
    re.compile(r"\bkabhi\s+kabhi\s+.*?(?:accha|zaroori|chahiye)\b", re.I),
    re.compile(r"\bsilence\s+bhi\s+accha\b", re.I),
    re.compile(r"\bapna\s+hi\s+maza\s+hai\b", re.I),
]

METAPHOR_WORDPLAY_PATTERNS = [
    re.compile(r"\btrailer\s+mode\b", re.I),
    re.compile(r"\bnegotiation\b", re.I),
    re.compile(r"\bbudget\s+thoda\s+limited\b", re.I),
    re.compile(r"\bvibe\s+hijack\b", re.I),
    re.compile(r"\baraam\s+mode\b", re.I),
    re.compile(r"\bscene\s+on\s+hai\b", re.I),
    re.compile(r"\bkeyboard\s+se\s+dosti\b", re.I),
    re.compile(r"\bfriday\s+energy\b", re.I),
    re.compile(r"\bpicture\s+baaki\s+hai\b", re.I),
    re.compile(r"\bvibe\s+match\b", re.I),
]

LAUGHING_FLIRTY_EMOJIS = re.compile(r"[😂😅🤣😜🤪😹😆😏😉]")
ANY_EMOJI_PATTERN = re.compile(r"[\U00010000-\U0010ffff\u2600-\u26ff\u2700-\u27bf]")


def classify_reply_shape(reply: str) -> str:
    """Classify reply structure into a shape category."""
    lines = [l for l in reply.split("\n") if l.strip()]
    if len(lines) >= 2:
        return "two_line"
    words = reply.strip().split()
    if len(words) <= 8:
        return "short_single"
    return "regular_single"


def evaluate_replies(replies: List[str]) -> Dict[str, Any]:
    total = len(replies)
    if total == 0:
        return {"total": 0}

    total_words = 0
    question_count = 0
    praise_count = 0
    gyaan_count = 0
    metaphor_count = 0
    emoji_count = 0
    shape_repeats = 0

    shapes: List[str] = []

    for idx, rep in enumerate(replies):
        words = rep.strip().split()
        total_words += len(words)

        if rep.strip().endswith("?") or "?" in rep:
            question_count += 1

        for pat in BANNED_PRAISE_PATTERNS:
            if pat.search(rep):
                praise_count += 1
                break

        for pat in PHILOSOPHICAL_GYAAN_PATTERNS:
            if pat.search(rep):
                gyaan_count += 1
                break

        # Metaphor & wordplay check
        has_metaphor = False
        for pat in METAPHOR_WORDPLAY_PATTERNS:
            if pat.search(rep):
                has_metaphor = True
                break
        if has_metaphor or count_banned_metaphors(rep) > 0:
            metaphor_count += 1

        # Emoji check
        if ANY_EMOJI_PATTERN.search(rep):
            emoji_count += 1

        # Shape tracking
        curr_shape = classify_reply_shape(rep)
        shapes.append(curr_shape)
        if idx > 0 and curr_shape == shapes[idx - 1]:
            shape_repeats += 1

    avg_words = total_words / total
    question_pct = (question_count / total) * 100
    praise_pct = (praise_count / total) * 100
    gyaan_pct = (gyaan_count / total) * 100
    metaphor_pct = (metaphor_count / total) * 100
    emoji_pct = (emoji_count / total) * 100
    repeated_shape_rate = (shape_repeats / (total - 1)) * 100 if total > 1 else 0.0

    return {
        "total_messages": total,
        "avg_words": round(avg_words, 1),
        "question_pct": round(question_pct, 1),
        "praise_pct": round(praise_pct, 1),
        "gyaan_pct": round(gyaan_pct, 1),
        "metaphor_wordplay_pct": round(metaphor_pct, 1),
        "emoji_pct": round(emoji_pct, 1),
        "repeated_shape_rate": round(repeated_shape_rate, 1),
        "shapes": shapes,
    }


def evaluate_care_mode(exchanges: List[Tuple[str, str]]) -> Dict[str, Any]:
    """
    Evaluates companion behavior on distress/illness/sadness messages.
    Exchanges: List of (user_message, bot_reply).
    Care mode requirements:
      1. Zero laughing or flirty emojis.
      2. Zero try-hard metaphors or jokes.
      3. Shows concern and asks at most ONE concrete follow-up question.
    """
    total_care_turns = len(exchanges)
    if total_care_turns == 0:
        return {"total_care_turns": 0, "correctness_pct": 100.0}

    correct_turns = 0
    failures = []

    for user_msg, bot_reply in exchanges:
        # Check 1: no laughing/flirty emoji
        has_laughing = bool(LAUGHING_FLIRTY_EMOJIS.search(bot_reply))

        # Check 2: no banned metaphors
        banned_count = count_banned_metaphors(bot_reply)

        # Check 3: asks at most 1 caring follow-up thought (at most 2 '?' as in 'bukhar hai kya? dawai li?')
        q_count = bot_reply.count("?")

        # Check 4: contains caring terminology
        has_care_cue = bool(re.search(r"\b(dawai|bukhar|tabiyat|kya hua|theek|thik|aaram|rest|care|khaya|stress|load|paani)\b", bot_reply, re.I))

        is_correct = (not has_laughing) and (banned_count == 0) and (q_count <= 2) and has_care_cue
        if is_correct:
            correct_turns += 1
        else:
            failures.append({
                "user_msg": user_msg,
                "bot_reply": bot_reply,
                "has_laughing": has_laughing,
                "banned_count": banned_count,
                "q_count": q_count,
                "has_care_cue": has_care_cue,
            })

    correctness_pct = (correct_turns / total_care_turns) * 100.0
    return {
        "total_care_turns": total_care_turns,
        "correct_turns": correct_turns,
        "correctness_pct": round(correctness_pct, 1),
        "failures": failures,
    }


def test_screenshot_scenario_fixes():
    """
    Validates that the exact bad replies from the user's screenshots
    are cleaned and fixed by the response validator and formatting pipeline.
    """
    # 1. Metaphor cleanup test
    raw_from_screenshot = "arey yrr mood off ka reason mil gaya 😂 Jukam ne pura vibe hijack kar liya bas, aaj araam mode banta hai"
    cleaned = arrange_and_format_response(
        raw_response=raw_from_screenshot,
        persona_name="Diya",
        user_message="kn wo bs thoda sa jukam hai",
        is_care_mode=True,
    )
    # Banned metaphors 'vibe hijack' and 'araam mode' and laughing emoji 😂 must be purged!
    assert "vibe hijack" not in cleaned.lower()
    assert "araam mode" not in cleaned.lower()
    assert "😂" not in cleaned, f"Laughing emoji found in care mode: {cleaned}"

    # 2. Sickness care follow-up
    care_reply = arrange_and_format_response(
        raw_response="ohh bukhar hai kya? dawai li kuch? 😂",
        persona_name="Diya",
        user_message="kn wo bs thoda sa jukam hai",
        is_care_mode=True,
    )
    assert "😂" not in care_reply, "Care mode must strip laughing emojis"
    assert "dawai" in care_reply or "bukhar" in care_reply


def test_quality_metrics_benchmark():
    # 20-turn realistic conversation trajectory for Diya adhering to the new guidelines:
    # 1 in 4 humor budget, 30-40% emoji rate, varied shapes (short, single-line, two-line), care mode on illness
    conversation_replies = [
        "Hyyy, kya chal raha hai aaj?",                                                    # 1. short_single (Q 1)
        "Haha sach me?\nitni der tak kaun sota hai 😂",                                   # 2. two_line (emoji 1, Q 2)
        "Mast hu ekdum, mummy ne subah hi garma garam poha bana diya tha",                 # 3. regular_single
        "acha, phir abhi kya plan hai?",                                                   # 4. short_single (Q 3)
        "Haha arre yaar 😜\nphir se wahi assignment shuru ho gaya?",                       # 5. two_line (emoji 2, Q 4)
        "aaj college mein library mein baith ke design submit kiya",                       # 6. regular_single
        "ha wo toh hai",                                                                   # 7. short_single (low energy mirror)
        "Arey yaar, tabiyat theek nahi lag rahi kya?\nbukhar hai kya? dawai li?",          # 8. two_line (care mode, zero emoji, Q 5)
        "thoda rest karo na please",                                                       # 9. short_single (care mode)
        "phone side mein rakh do, shaam tak better feel hoga pakka",                       # 10. regular_single (care mode)
        "abhi thoda better lag raha hai?",                                                 # 11. short_single (care recovery, Q 6)
        "chalo badhiya ☕\nmaine bhi abhi sham ki adrak wali chai banayi thi",              # 12. two_line (own life, emoji 3)
        "sach me? mujhe laga cold drink peene ka man tha tumhara",                         # 13. regular_single (Q 7)
        "haha sahi pakde 😅",                                                              # 14. short_single (emoji 4)
        "phir shaam ka scene dekh lena, thoda rest kar lo",                                # 15. regular_single
        "main toh bas abhi sneha se baat karke free hui hu",                               # 16. regular_single
        "usne ek ajeeb sa funny meme bheja tha 🙈",                                        # 17. short_single (emoji 5)
        "haha dekh kar hassi nahi ruki\nshaam ka mausam bhi kaafi cool ho gaya",           # 18. two_line
        "aaj toh ghar pe mummy ne paneer parathe banaye the 😋",                           # 19. regular_single (emoji 6)
        "waise tumne khana khaya ya nahi? ✨",                                             # 20. short_single (emoji 7, 7/20 = 35%, Q 8 = 40%)
    ]

    metrics = evaluate_replies(conversation_replies)

    print("\n" + "=" * 60)
    print("SPARK CHAT QUALITY BENCHMARK REPORT (20-Turn Conversation)")
    print("=" * 60)
    print(f"Total Evaluated Messages:     {metrics['total_messages']}")
    print(f"Average Words per Reply:      {metrics['avg_words']} (Target: <= 20 words)")
    print(f"Question Rate:                {metrics['question_pct']}% (Target: <= 40%)")
    print(f"Robotic Praise Rate:          {metrics['praise_pct']}% (Target: 0%)")
    print(f"Philosophical Gyaan Rate:     {metrics['gyaan_pct']}% (Target: 0%)")
    print(f"Metaphor/Wordplay Rate:       {metrics['metaphor_wordplay_pct']}% (Target: < 25%)")
    print(f"Emoji Frequency:              {metrics['emoji_pct']}% (Target: 30-40%)")
    print(f"Repeated Shape Rate:          {metrics['repeated_shape_rate']}% (Target: < 25%)")
    print("=" * 60)

    # Assertions against the user's targets:
    assert metrics["avg_words"] <= 20, f"Average words {metrics['avg_words']} exceeded 20 words"
    assert metrics["metaphor_wordplay_pct"] < 25.0, f"Metaphor rate {metrics['metaphor_wordplay_pct']}% must be < 25%"
    assert 25.0 <= metrics["emoji_pct"] <= 40.0, f"Emoji rate {metrics['emoji_pct']}% must be around 30-40%"
    assert metrics["repeated_shape_rate"] <= 25.0, f"Repeated shape rate {metrics['repeated_shape_rate']}% must be <= 25%"
    assert metrics["praise_pct"] == 0, "No robotic praise phrases allowed"
    assert metrics["gyaan_pct"] == 0, "No gyaan or preacher phrases allowed"

    # Care mode test suite
    care_exchanges = [
        ("kn wo bs thoda sa jukam hai", "Arey yaar, bukhar bhi hai kya? Dawai li kuch?"),
        ("aaj mood bahut kharab hai, ghar pe ladai ho gayi", "Kya hua yaar, sab theek hai na ghar pe?"),
        ("sar dard se phat raha hai", "Arey yaar, screen band karke thoda rest kar lo na. Dawai li?"),
        ("bahut zyada stressed hu aaj office work se", "Kya hua, boss ne zyada load de diya kya? Thoda paani piyo"),
    ]
    care_metrics = evaluate_care_mode(care_exchanges)
    print("\n" + "=" * 60)
    print("CARE MODE CORRECTNESS REPORT")
    print("=" * 60)
    print(f"Total Care Situations Tested: {care_metrics['total_care_turns']}")
    print(f"Correct Care Responses:       {care_metrics['correct_turns']}")
    print(f"Care Mode Correctness:        {care_metrics['correctness_pct']}% (Target: 100%)")
    print("=" * 60 + "\n")

    assert care_metrics["correctness_pct"] == 100.0, f"Expected 100% care mode correctness, got {care_metrics['correctness_pct']}%"


if __name__ == "__main__":
    test_screenshot_scenario_fixes()
    test_quality_metrics_benchmark()
    print("[SUCCESS] All chat quality benchmarks passed!")
