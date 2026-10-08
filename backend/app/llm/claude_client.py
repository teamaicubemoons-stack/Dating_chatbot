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

import anthropic
from anthropic import APIStatusError, RateLimitError

from app.llm.base_llm_interface import BaseLLMClient
from app.config.settings import settings
from app.utils.error_handler import LLMClientError

logger = logging.getLogger(__name__)


class ClaudeClient(BaseLLMClient):
    """Anthropic Claude inference via the official anthropic-python SDK."""

    def __init__(self) -> None:
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._model = settings.claude_model
        logger.info("ClaudeClient initialised — model: %s", self._model)

    def generate_response(
        self,
        system_prompt: str,
        chat_history: List[Dict[str, str]],
        user_message: str,
    ) -> str:
        """
        Call the Anthropic Messages API.
        Claude takes system prompt separately from the messages array.
        """
        # Build messages list (no system message in the array for Claude)
        messages = list(chat_history)  # copy — don't mutate caller's list
        messages.append({"role": "user", "content": user_message})

        try:
            response = self._client.messages.create(
                model=self._model,
                system=system_prompt,
                messages=messages,
                temperature=0.85,
                max_tokens=512,
            )
            reply = response.content[0].text.strip()
            logger.debug("Claude response received (%d chars)", len(reply))
            return reply

        except (RateLimitError, APIStatusError) as exc:
            logger.error("Claude API error: %s", exc)
            raise LLMClientError(f"Claude error: {exc}") from exc
        except Exception as exc:
            logger.error("Unexpected Claude error: %s", exc)
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
