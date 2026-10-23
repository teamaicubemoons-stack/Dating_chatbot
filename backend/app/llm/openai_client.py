"""
llm/openai_client.py
--------------------
OpenAI-only inference client for gpt-5.6-luna (primary) and gpt-5.6-terra (fallback).

Confirmed parameter support on 2026-10-23 via live API probe:
  - max_completion_tokens  ✅  (max_tokens → 400 BadRequest)
  - reasoning_effort       ✅  "none" | "low" | "medium" | "high"
  - temperature            ✅  accepted alongside reasoning_effort
  - top_p                  ✅  accepted

Design:
  - Primary model: gpt-5.6-luna  (reasoning_effort=none, fastest)
  - Fallback model: gpt-5.6-terra (same params, triggered on timeout/429/5xx)
  - max_completion_tokens=500 gives room for 1-3 short text lines
  - Token usage (input / cached / output) + estimated USD cost logged per request
  - Daily cost accumulator printed at midnight or on demand
  - Per-user rate limiting and max input length enforced here
"""

from __future__ import annotations

import logging
import time
import threading
from collections import defaultdict
from datetime import datetime, date
from typing import List, Dict, Optional

import openai
from openai import (
    APIStatusError,
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    BadRequestError,
)

from app.llm.base_llm_interface import BaseLLMClient
from app.config.settings import settings
from app.utils.error_handler import LLMClientError

logger = logging.getLogger(__name__)

# ─── Cost accumulator (thread-safe) ──────────────────────────────────────────

_cost_lock = threading.Lock()
_daily_cost: Dict[str, float] = {"date": str(date.today()), "usd": 0.0}
_daily_requests: int = 0
_daily_input_tokens: int = 0
_daily_output_tokens: int = 0
_daily_cached_tokens: int = 0
_regen_total: int = 0
_regen_count: int = 0


def _accrue_cost(input_tok: int, cached_tok: int, output_tok: int, model: str) -> float:
    """Add to daily cost accumulator and return this request's cost in USD."""
    # Use luna pricing as default; terra is slightly more expensive but we keep one rate
    cost_in  = (input_tok - cached_tok) * settings.cost_per_million_input_tokens / 1_000_000
    cost_cached = cached_tok * (settings.cost_per_million_input_tokens * 0.25) / 1_000_000
    cost_out = output_tok * settings.cost_per_million_output_tokens / 1_000_000
    total = cost_in + cost_cached + cost_out

    global _daily_cost, _daily_requests, _daily_input_tokens, _daily_output_tokens, _daily_cached_tokens
    with _cost_lock:
        today = str(date.today())
        if _daily_cost["date"] != today:
            # New day: print yesterday's summary and reset
            _print_daily_summary()
            _daily_cost = {"date": today, "usd": 0.0}
            _daily_requests = 0
            _daily_input_tokens = 0
            _daily_output_tokens = 0
            _daily_cached_tokens = 0
        _daily_cost["usd"] += total
        _daily_requests += 1
        _daily_input_tokens += input_tok
        _daily_output_tokens += output_tok
        _daily_cached_tokens += cached_tok
    return total


def _print_daily_summary() -> None:
    regen_rate = (_regen_count / _regen_total * 100) if _regen_total else 0.0
    summary = (
        "\n" + "=" * 65 + "\n"
        " [DAILY COST SUMMARY]\n"
        f"  Date           : {_daily_cost['date']}\n"
        f"  Requests       : {_daily_requests}\n"
        f"  Input Tokens   : {_daily_input_tokens} ({_daily_cached_tokens} cached)\n"
        f"  Output Tokens  : {_daily_output_tokens}\n"
        f"  Estimated USD  : ${_daily_cost['usd']:.4f}\n"
        f"  Regen Rate     : {regen_rate:.1f}% ({_regen_count}/{_regen_total})\n"
        + "=" * 65 + "\n"
    )
    print(summary, flush=True)
    logger.info("Daily cost summary: $%.4f over %d requests", _daily_cost["usd"], _daily_requests)


def log_daily_summary() -> None:
    """Public hook — call from scheduler or /health endpoint."""
    with _cost_lock:
        _print_daily_summary()


def record_regeneration(triggered: bool) -> None:
    """Called by the post-processor to track regeneration rate."""
    global _regen_total, _regen_count
    with _cost_lock:
        _regen_total += 1
        if triggered:
            _regen_count += 1


# ─── Per-user rate limiter ───────────────────────────────────────────────────

_rate_lock = threading.Lock()
# {user_id: [timestamp, ...]}  — sliding window
_user_request_times: Dict[str, list] = defaultdict(list)


def _check_rate_limit(user_id: Optional[str]) -> None:
    """Raise LLMClientError if user exceeds rate limit."""
    if not user_id:
        return
    limit = settings.user_rate_limit_per_minute
    now = time.monotonic()
    window = 60.0
    with _rate_lock:
        times = _user_request_times[user_id]
        times = [t for t in times if now - t < window]
        _user_request_times[user_id] = times
        if len(times) >= limit:
            raise LLMClientError(
                f"Rate limit: user {user_id[:8]} sent {limit}+ messages in 60s"
            )
        times.append(now)


# ─── OpenAI Client ───────────────────────────────────────────────────────────

class OpenAIClient(BaseLLMClient):
    """
    OpenAI inference client.
    Primary: settings.chat_model (gpt-5.6-luna)
    Fallback: settings.fallback_model (gpt-5.6-terra)
    Both use reasoning_effort=none, temperature=0.95, max_completion_tokens=500.
    """

    def __init__(self) -> None:
        self._client = openai.OpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.llm_request_timeout_seconds,
            max_retries=0,  # We do manual fallback
        )
        self._primary  = settings.chat_model     or "gpt-5.6-luna"
        self._fallback = settings.fallback_model or "gpt-5.6-terra"
        self._reasoning_effort = settings.chat_reasoning_effort  # "none"
        self._max_tokens = settings.chat_max_completion_tokens   # 500
        self._temperature = settings.chat_temperature            # 0.95
        self._top_p = settings.chat_top_p                       # 1.0

        logger.info(
            "OpenAIClient ready — primary: %s | fallback: %s | "
            "reasoning_effort: %s | max_completion_tokens: %d | temperature: %.2f",
            self._primary, self._fallback,
            self._reasoning_effort, self._max_tokens, self._temperature,
        )

    # ── internal ──────────────────────────────────────────────────────────────

    def _build_messages(
        self,
        system_prompt: str,
        chat_history: List[Dict[str, str]],
        user_message: str,
    ) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        history = list(chat_history or [])
        # Drop duplicate current-turn user message if caller already appended it
        if (
            history
            and history[-1].get("role") == "user"
            and (history[-1].get("content") or "").strip() == (user_message or "").strip()
        ):
            history.pop()
        for m in history:
            c = (m.get("content") or "").strip()
            if c and m.get("role") in {"user", "assistant"}:
                messages.append({"role": m["role"], "content": c})
        messages.append({"role": "user", "content": user_message})
        return messages

    def _call(self, model: str, messages: List[Dict]) -> tuple[str, object]:
        """Single API call. Returns (text, usage)."""
        # Build kwargs — only pass params confirmed to work on gpt-5.6 family
        kwargs: Dict = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": self._max_tokens,
            "reasoning_effort": self._reasoning_effort,
            "temperature": self._temperature,
        }
        if self._top_p < 1.0:
            kwargs["top_p"] = self._top_p

        resp = self._client.chat.completions.create(**kwargs)
        text = (resp.choices[0].message.content if resp.choices else "") or ""
        return text.strip(), resp.usage

    def _log_metrics(
        self,
        model: str,
        elapsed: float,
        usage,
        is_fallback: bool,
    ) -> None:
        try:
            input_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
        except (TypeError, ValueError):
            input_tok = 0
        try:
            output_tok = int(getattr(usage, "completion_tokens", 0) or 0)
        except (TypeError, ValueError):
            output_tok = 0
        try:
            total_tok = int(getattr(usage, "total_tokens", input_tok + output_tok) or (input_tok + output_tok))
        except (TypeError, ValueError):
            total_tok = input_tok + output_tok

        details = getattr(usage, "prompt_tokens_details", None)
        try:
            cached_tok = int(getattr(details, "cached_tokens", 0) or 0) if details else 0
        except (TypeError, ValueError):
            cached_tok = 0

        cost_usd = float(_accrue_cost(input_tok, cached_tok, output_tok, model))

        tag = " [FALLBACK]" if is_fallback else ""
        metrics = (
            "\n" + "=" * 65 + "\n"
            f" [LLM RESPONSE METRICS{tag}]\n"
            f"  Model          : {model}\n"
            f"  Latency        : {elapsed:.2f}s\n"
            f"  Input Tokens   : {input_tok} ({cached_tok} cached)\n"
            f"  Output Tokens  : {output_tok}\n"
            f"  Total Tokens   : {total_tok}\n"
            f"  Est. Cost      : ${cost_usd:.5f}\n"
            f"  Daily Total    : ${_daily_cost['usd']:.4f}\n"
            + "=" * 65 + "\n"
        )
        print(metrics, flush=True)
        logger.info(
            "OpenAI | model=%s | latency=%.2fs | in=%d cached=%d out=%d | $%.5f%s",
            model, elapsed, input_tok, cached_tok, output_tok, cost_usd,
            " [fallback]" if is_fallback else "",
        )

    # ── public API ────────────────────────────────────────────────────────────

    def generate_response(
        self,
        system_prompt: str,
        chat_history: List[Dict[str, str]],
        user_message: str,
        user_id: Optional[str] = None,
    ) -> str:
        # 1. Rate-limit check
        _check_rate_limit(user_id)

        # 2. Input length guard
        max_len = settings.max_user_input_length
        if len(user_message) > max_len:
            logger.warning("Input truncated: %d → %d chars", len(user_message), max_len)
            user_message = user_message[:max_len]

        messages = self._build_messages(system_prompt, chat_history, user_message)

        # 3. Primary call (gpt-5.6-luna)
        t0 = time.perf_counter()
        try:
            text, usage = self._call(self._primary, messages)
            elapsed = time.perf_counter() - t0
            self._log_metrics(self._primary, elapsed, usage, is_fallback=False)
            if not text:
                raise LLMClientError("OpenAI primary returned empty response")
            return text
        except (RateLimitError, APITimeoutError, APIConnectionError) as exc:
            elapsed = time.perf_counter() - t0
            logger.warning(
                "Primary %s failed after %.2fs (%s); trying fallback %s",
                self._primary, elapsed, type(exc).__name__, self._fallback,
            )
        except APIStatusError as exc:
            elapsed = time.perf_counter() - t0
            if exc.status_code >= 500:
                logger.warning(
                    "Primary %s 5xx after %.2fs; trying fallback %s",
                    self._primary, elapsed, self._fallback,
                )
            else:
                raise LLMClientError(f"OpenAI API error {exc.status_code}: {exc}") from exc
        except BadRequestError as exc:
            raise LLMClientError(f"OpenAI bad request: {exc}") from exc
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            logger.error("Unexpected error on primary after %.2fs: %s", elapsed, exc)
            raise LLMClientError(f"OpenAI unexpected: {exc}") from exc

        # 4. Fallback call (gpt-5.6-terra) — same system prompt + live context
        t0 = time.perf_counter()
        try:
            text, usage = self._call(self._fallback, messages)
            elapsed = time.perf_counter() - t0
            self._log_metrics(self._fallback, elapsed, usage, is_fallback=True)
            if not text:
                raise LLMClientError("OpenAI fallback returned empty response")
            return text
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            logger.error("Fallback %s also failed after %.2fs: %s", self._fallback, elapsed, exc)
            raise LLMClientError(f"Both OpenAI models failed: {exc}") from exc

    def health_check(self) -> bool:
        """
        Verify OpenAI is reachable using a free models list call
        (no chat completion, no token cost).
        """
        try:
            self._client.models.list()
            return True
        except Exception as exc:
            logger.warning("OpenAI health check (models.list) failed: %s", exc)
            return False
