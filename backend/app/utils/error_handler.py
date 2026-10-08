"""
utils/error_handler.py
-----------------------
App-level exception hierarchy used across modules to decouple
provider-specific errors from routing/orchestration logic.

Adding a new provider = raise one of these exceptions from its client wrapper.
llm_router.py catches these — it never imports provider-specific exception classes.
"""


class AppError(Exception):
    """Base class for all application-level exceptions."""


class GroqQuotaExceededError(AppError):
    """
    Raised by GroqClient when the API returns a rate-limit / quota-exhausted
    response. Caught by llm_router.py to trigger silent Claude fallback.
    """


class LLMClientError(AppError):
    """
    Generic LLM call failure (network error, bad response, unexpected 5xx, etc.).
    Raised when the error is NOT a recoverable quota issue.
    """


class PersonaNotFoundError(AppError):
    """Raised when a requested persona ID does not exist in configs."""


class RAGError(AppError):
    """Raised for ChromaDB / embedding failures that shouldn't crash the app."""


class MemoryError(AppError):
    """Raised for DB-layer memory persistence failures."""
