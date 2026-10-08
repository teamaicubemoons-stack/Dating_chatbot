"""
llm/base_llm_interface.py
--------------------------
Abstract base class that BOTH groq_client.py and claude_client.py must implement.
llm_router.py only ever calls this interface — it never imports provider-specific code.

This decouples the routing logic from provider implementation details completely.
Adding a new provider (e.g., OpenAI) = implement this class + register in llm_router.
"""

from abc import ABC, abstractmethod
from typing import List, Dict


class BaseLLMClient(ABC):
    """
    Common interface for all LLM provider wrappers.

    Both `generate_response` signatures must match exactly so llm_router.py
    can call either interchangeably without branching on provider details.
    """

    @abstractmethod
    def generate_response(
        self,
        system_prompt: str,
        chat_history: List[Dict[str, str]],
        user_message: str,
    ) -> str:
        """
        Generate an AI reply given a fully-built system prompt, prior chat
        history, and the latest user message.

        Args:
            system_prompt:  The fully constructed persona + context prompt.
            chat_history:   List of {"role": "user"|"assistant", "content": str}
                            representing prior turns (most recent last).
            user_message:   The current user message text.

        Returns:
            The AI-generated reply string.

        Raises:
            RateLimitError / QuotaExceededError: provider-specific subclasses
            should re-raise these as the app-level exceptions defined in
            utils/error_handler.py so llm_router.py can catch them uniformly.
        """
        ...

    @abstractmethod
    def health_check(self) -> bool:
        """
        Send a minimal test call to verify the provider is reachable and
        the API key/quota is valid.

        Returns:
            True  — provider is healthy and has available quota.
            False — provider is unavailable or quota is exhausted.
        """
        ...
