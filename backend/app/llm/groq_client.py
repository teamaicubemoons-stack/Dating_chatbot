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

from groq import Groq, RateLimitError, APIStatusError

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
    """Groq LLaMA inference via the official groq-python SDK."""

    def __init__(self) -> None:
        self._client = Groq(api_key=settings.groq_api_key)
        self._model = settings.groq_model
        logger.info("GroqClient initialised — model: %s", self._model)

    def generate_response(
        self,
        system_prompt: str,
        chat_history: List[Dict[str, str]],
        user_message: str,
    ) -> str:
        """Call Groq chat completions. Raises GroqQuotaExceededError on quota errors."""
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(chat_history)
        messages.append({"role": "user", "content": user_message})

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0.85,        # warm, varied responses
                max_tokens=512,
                top_p=0.95,
            )
            reply = response.choices[0].message.content.strip()
            logger.debug("Groq response received (%d chars)", len(reply))
            return reply

        except Exception as exc:
            if _is_quota_error(exc):
                logger.warning("Groq quota exceeded — triggering fallback: %s", exc)
                raise GroqQuotaExceededError(str(exc)) from exc
            logger.error("Groq API error: %s", exc)
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
