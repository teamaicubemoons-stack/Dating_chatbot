"""
tests/model_ab_test.py
----------------------
A/B probe comparing gpt-5.6-luna (primary) vs gpt-5.6-terra (fallback)
on latency, token efficiency, and response style.
Run standalone: python -m tests.model_ab_test
"""

import time
import os
import openai
from app.config.settings import settings

PROMPT = [
    {"role": "system", "content": "You are Diya, 21, living in Raipur. You are texting your dating match. Keep it casual, natural Hinglish, 1 short line, witty, tease them playfully."},
    {"role": "user", "content": "mujhe chai bilkul pasand nahi, only cold brew"}
]


def probe_model(model_name: str):
    client = openai.OpenAI(api_key=settings.openai_api_key)
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model_name,
        messages=PROMPT,
        temperature=0.95,
        max_completion_tokens=150,
        reasoning_effort="none",
    )
    latency = time.perf_counter() - t0
    reply = resp.choices[0].message.content
    usage = resp.usage
    return {
        "model": model_name,
        "latency_sec": round(latency, 2),
        "reply": reply,
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
    }


def run_ab_comparison():
    print(f"\n--- Running A/B Probe ({settings.chat_model} vs {settings.fallback_model}) ---")
    try:
        luna_result = probe_model(settings.chat_model)
        print(f"[{luna_result['model']}] Latency: {luna_result['latency_sec']}s | Tokens: in={luna_result['input_tokens']} out={luna_result['output_tokens']}")
        print(f"Reply: {luna_result['reply']}\n")
    except Exception as e:
        print(f"Luna probe failed: {e}")

    try:
        terra_result = probe_model(settings.fallback_model)
        print(f"[{terra_result['model']}] Latency: {terra_result['latency_sec']}s | Tokens: in={terra_result['input_tokens']} out={terra_result['output_tokens']}")
        print(f"Reply: {terra_result['reply']}\n")
    except Exception as e:
        print(f"Terra probe failed: {e}")


if __name__ == "__main__":
    run_ab_comparison()
