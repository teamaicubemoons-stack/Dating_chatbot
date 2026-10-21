"""
llm/openai_client.py
--------------------
OpenAI API wrapper. Implements BaseLLMClient.

Responsibility:
  - Make chat completion calls to OpenAI (e.g. gpt-4o-mini).
  - Act as the reliable fallback when Groq quota is exhausted.
  - Never contain routing logic.
"""

import logging
from typing import List, Dict
import time

import openai
from openai import APIStatusError, RateLimitError, APITimeoutError, APIConnectionError

from app.llm.base_llm_interface import BaseLLMClient
from app.config.settings import settings
from app.utils.error_handler import LLMClientError

logger = logging.getLogger(__name__)


class OpenAIClient(BaseLLMClient):
    """OpenAI inference via the official openai SDK with bounded timeouts."""

    def __init__(self) -> None:
        timeout_sec = getattr(settings, "llm_request_timeout_seconds", 12.0)
        retries = getattr(settings, "llm_max_retries", 1)
        self._client = openai.OpenAI(
            api_key=settings.openai_api_key,
            timeout=timeout_sec,
            max_retries=retries,
        )
        self._model = getattr(settings, "openai_model", "gpt-4o-mini")
        logger.info(
            "OpenAIClient initialised — model: %s, timeout: %.1fs, max_retries: %d",
            self._model, timeout_sec, retries
        )

    def generate_response(
        self,
        system_prompt: str,
        chat_history: List[Dict[str, str]],
        user_message: str,
    ) -> str:
        """
        Call OpenAI Chat Completions API with bounded timeout.
        System prompt is passed as the first message with role='system'.
        """
        messages = [{"role": "system", "content": system_prompt}]
        history = list(chat_history or [])
        # The current turn is appended below. Avoid duplicating it when a
        # caller passes a history snapshot taken after persistence.
        if history and history[-1].get("role") == "user" and (
            history[-1].get("content") or "").strip() == (user_message or "").strip():
            history.pop()
        for m in history:
            c = (m.get("content") or "").strip()
            if c and m.get("role") in {"user", "assistant"}:
                messages.append({"role": m["role"], "content": c})
        messages.append({"role": "user", "content": user_message})

        t0 = time.perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0.95,
                presence_penalty=0.4,
                frequency_penalty=0.3,
                max_tokens=300,
            )
            elapsed = time.perf_counter() - t0
            raw_text = response.choices[0].message.content if response.choices else ""
            reply = (raw_text or "").strip()

            usage = getattr(response, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
            completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
            total_tokens = getattr(usage, "total_tokens", prompt_tokens + completion_tokens) if usage else (prompt_tokens + completion_tokens)

            # Prominently display LLM response metrics in terminal
            metrics_display = (
                "\n" + "=" * 65 + "\n"
                " [LLM RESPONSE METRICS]\n"
                f"  Provider       : OpenAI\n"
                f"  Model          : {self._model}\n"
                f"  Latency        : {elapsed:.2f}s\n"
                f"  Input Tokens   : {prompt_tokens} (prompt)\n"
                f"  Output Tokens  : {completion_tokens} (completion)\n"
                f"  Total Tokens   : {total_tokens}\n"
                + "=" * 65 + "\n"
            )
            print(metrics_display, flush=True)

            if not reply:
                logger.warning("OpenAI returned empty response (0 chars) after %.2fs", elapsed)
                raise LLMClientError("OpenAI returned empty response")

            logger.info(
                "OpenAI Metrics | Model: %s | Latency: %.2fs | In: %d | Out: %d | Total: %d",
                self._model, elapsed, prompt_tokens, completion_tokens, total_tokens
            )
            return reply

        except (RateLimitError, APIStatusError) as exc:
            elapsed = time.perf_counter() - t0
            logger.error("OpenAI API error after %.2fs: %s", elapsed, exc)
            raise LLMClientError(f"OpenAI error: {exc}") from exc
        except (APITimeoutError, APIConnectionError) as exc:
            elapsed = time.perf_counter() - t0
            logger.warning("OpenAI timeout/connection stall after %.2fs: %s", elapsed, exc)
            raise LLMClientError(f"OpenAI timeout after {elapsed:.1f}s: {exc}") from exc
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            logger.error("Unexpected OpenAI error after %.2fs: %s", elapsed, exc)
            raise LLMClientError(f"OpenAI unexpected error: {exc}") from exc

    def health_check(self) -> bool:
        """Verify that OpenAI is reachable with active credentials."""
        try:
            self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            return True
        except Exception as exc:
            logger.warning("OpenAI health check failed: %s", exc)
            return False
