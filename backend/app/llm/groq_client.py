"""
llm/groq_client.py
------------------
Groq API wrapper. Implements BaseLLMClient.

Responsibility:
  - Make chat completion calls to Groq.
  - Detect quota/rate-limit errors and raise GroqQuotaExceededError
    so llm_router.py can trigger the fallback to Claude.
  - Never contain routing logic — that lives solely in llm_router.py.
"""

import logging
from typing import List, Dict

import time
from groq import Groq, RateLimitError, APIStatusError, APITimeoutError, APIConnectionError

from app.llm.base_llm_interface import BaseLLMClient
from app.config.settings import settings
from app.utils.error_handler import GroqQuotaExceededError, LLMClientError

logger = logging.getLogger(__name__)

# Groq error codes / messages that indicate daily/monthly quota exhaustion
_QUOTA_STATUS_CODES = {429}
_QUOTA_ERROR_TYPES = {"tokens", "requests", "rate_limit_exceeded", "insufficient_quota"}


def _is_quota_error(e: Exception) -> bool:
    """Return True if the exception indicates Groq quota/rate-limit exhaustion."""
    if isinstance(e, RateLimitError):
        return True
    if isinstance(e, APIStatusError):
        if e.status_code in _QUOTA_STATUS_CODES:
            return True
        # Check error body for quota-specific messages
        body = str(e).lower()
        return any(kw in body for kw in _QUOTA_ERROR_TYPES)
    return False


class GroqClient(BaseLLMClient):
    """Groq LLaMA inference via the official groq-python SDK with bounded timeouts."""

    def __init__(self) -> None:
        timeout_sec = getattr(settings, "llm_request_timeout_seconds", 12.0)
        retries = getattr(settings, "llm_max_retries", 1)
        self._client = Groq(
            api_key=settings.groq_api_key,
            timeout=timeout_sec,
            max_retries=retries,
        )
        self._model = settings.groq_model
        logger.info(
            "GroqClient initialised — model: %s, timeout: %.1fs, max_retries: %d",
            self._model, timeout_sec, retries
        )

    def generate_response(
        self,
        system_prompt: str,
        chat_history: List[Dict[str, str]],
        user_message: str,
    ) -> str:
        """Call Groq chat completions with bounded latency. Raises GroqQuotaExceededError on quota errors."""
        messages = [{"role": "system", "content": system_prompt}]
        for m in chat_history:
            c = (m.get("content") or "").strip()
            if c:
                messages.append({"role": m["role"], "content": c})
        messages.append({"role": "user", "content": user_message})

        t0 = time.perf_counter()
        for attempt in range(2):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=0.85,        # warm, varied responses
                    max_tokens=512,
                    top_p=0.95,
                )
                elapsed = time.perf_counter() - t0
                raw_content = response.choices[0].message.content
                reply = (raw_content or "").strip()

                usage = getattr(response, "usage", None)
                prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
                completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
                total_tokens = getattr(usage, "total_tokens", prompt_tokens + completion_tokens) if usage else (prompt_tokens + completion_tokens)

                # Prominently display LLM response metrics in terminal
                metrics_display = (
                    "\n" + "=" * 65 + "\n"
                    " [LLM RESPONSE METRICS]\n"
                    f"  Provider       : Groq\n"
                    f"  Model          : {self._model}\n"
                    f"  Latency        : {elapsed:.2f}s\n"
                    f"  Input Tokens   : {prompt_tokens} (prompt)\n"
                    f"  Output Tokens  : {completion_tokens} (completion)\n"
                    f"  Total Tokens   : {total_tokens}\n"
                    + "=" * 65 + "\n"
                )
                print(metrics_display, flush=True)

                if not reply:
                    logger.warning(
                        "Groq returned empty response (0 chars) after %.2fs — triggering fallback",
                        elapsed
                    )
                    raise LLMClientError("Groq returned empty response")

                logger.info(
                    "Groq Metrics | Model: %s | Latency: %.2fs | In: %d | Out: %d | Total: %d",
                    self._model, elapsed, prompt_tokens, completion_tokens, total_tokens
                )
                return reply

            except Exception as exc:
                elapsed = time.perf_counter() - t0
                # If short rate limit window (e.g. 'try again in X seconds') and first attempt, wait briefly and retry
                if attempt == 0 and _is_quota_error(exc) and "try again in" in str(exc).lower():
                    logger.warning("Groq rate limit burst window reached; sleeping 2.5s before retry: %s", exc)
                    time.sleep(2.5)
                    continue

                if _is_quota_error(exc):
                    logger.warning("Groq quota exceeded in %.2fs — triggering fallback: %s", elapsed, exc)
                    raise GroqQuotaExceededError(str(exc)) from exc
                if isinstance(exc, (APITimeoutError, APIConnectionError)):
                    logger.warning("Groq network/timeout stall after %.2fs — triggering fallback: %s", elapsed, exc)
                    raise LLMClientError(f"Groq timeout after {elapsed:.1f}s: {exc}") from exc
                if isinstance(exc, LLMClientError):
                    raise
                logger.error("Groq API error after %.2fs: %s", elapsed, exc)
                raise LLMClientError(f"Groq error: {exc}") from exc

    def health_check(self) -> bool:
        """
        Send a tiny probe message to Groq to test quota availability.
        Returns True if Groq responds successfully.
        """
        try:
            self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            logger.info("Groq health check: PASS")
            return True
        except Exception as exc:
            if _is_quota_error(exc):
                logger.info("Groq health check: FAIL (quota still exhausted)")
            else:
                logger.warning("Groq health check: FAIL (unexpected error: %s)", exc)
            return False
