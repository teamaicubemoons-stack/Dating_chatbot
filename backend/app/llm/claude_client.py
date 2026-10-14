"""
llm/claude_client.py
--------------------
Anthropic Claude API wrapper. Implements BaseLLMClient.

Responsibility:
  - Make chat completion calls to Claude (Messages API).
  - Act as the reliable fallback when Groq quota is exhausted.
  - Never contain routing logic.
"""

import logging
from typing import List, Dict

import time
import anthropic
from anthropic import APIStatusError, RateLimitError, APITimeoutError, APIConnectionError

from app.llm.base_llm_interface import BaseLLMClient
from app.config.settings import settings
from app.utils.error_handler import LLMClientError

logger = logging.getLogger(__name__)


class ClaudeClient(BaseLLMClient):
    """Anthropic Claude inference via the official anthropic-python SDK with bounded timeouts."""

    def __init__(self) -> None:
        timeout_sec = getattr(settings, "llm_request_timeout_seconds", 12.0)
        retries = getattr(settings, "llm_max_retries", 1)
        self._client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key,
            timeout=timeout_sec,
            max_retries=retries,
        )
        self._model = settings.claude_model
        logger.info(
            "ClaudeClient initialised — model: %s, timeout: %.1fs, max_retries: %d",
            self._model, timeout_sec, retries
        )

    def generate_response(
        self,
        system_prompt: str,
        chat_history: List[Dict[str, str]],
        user_message: str,
    ) -> str:
        """
        Call the Anthropic Messages API with bounded timeout.
        Claude takes system prompt separately from the messages array.
        """
        # Build messages list (no system message in the array for Claude)
        messages = []
        for m in chat_history:
            c = (m.get("content") or "").strip()
            if c:
                messages.append({"role": m["role"], "content": c})
        messages.append({"role": "user", "content": user_message})

        t0 = time.perf_counter()
        try:
            response = self._client.messages.create(
                model=self._model,
                system=system_prompt,
                messages=messages,
                temperature=0.85,
                max_tokens=512,
            )
            elapsed = time.perf_counter() - t0
            raw_text = response.content[0].text if response.content else ""
            reply = (raw_text or "").strip()

            if not reply:
                logger.warning("Claude returned empty response (0 chars) after %.2fs", elapsed)
                raise LLMClientError("Claude returned empty response")

            logger.info("Claude response received in %.2fs (%d chars)", elapsed, len(reply))
            return reply

        except (RateLimitError, APIStatusError) as exc:
            elapsed = time.perf_counter() - t0
            logger.error("Claude API error after %.2fs: %s", elapsed, exc)
            raise LLMClientError(f"Claude error: {exc}") from exc
        except (APITimeoutError, APIConnectionError) as exc:
            elapsed = time.perf_counter() - t0
            logger.warning("Claude timeout/connection stall after %.2fs: %s", elapsed, exc)
            raise LLMClientError(f"Claude timeout after {elapsed:.1f}s: {exc}") from exc
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            logger.error("Unexpected Claude error after %.2fs: %s", elapsed, exc)
            raise LLMClientError(f"Claude unexpected error: {exc}") from exc

    def health_check(self) -> bool:
        """Claude is our fallback — always assumed reachable if key is valid."""
        try:
            self._client.messages.create(
                model=self._model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            return True
        except Exception as exc:
            logger.warning("Claude health check failed: %s", exc)
            return False
