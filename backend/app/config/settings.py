"""
config/settings.py
------------------
Central configuration module. Loads all environment variables via pydantic-settings.
Import `settings` anywhere in the app — never read os.environ directly elsewhere.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    # --- LLM API Keys ---
    openai_api_key: str = ""

    # Legacy keys kept as optional so old .env files don't break on startup.
    # These are NOT used by any routing logic; only openai_api_key is active.
    groq_api_key: str = ""
    anthropic_api_key: str = ""

    # --- Active Model Names (OpenAI only) ---
    # CHAT_MODEL: primary fast model for all chat replies
    chat_model: str = "gpt-5.6-luna"
    # CHAT_MODEL_PREMIUM: used only for A/B testing or explicit upgrade
    chat_model_premium: str = "gpt-5.6-terra"
    # FALLBACK_MODEL: used ONLY if the primary call fails (timeout / rate-limit / 5xx)
    fallback_model: str = "gpt-5.6-terra"

    # --- Legacy model name fields (ignored by routing, kept for .env backward compat) ---
    groq_model: str = ""
    claude_model: str = ""
    openai_model: str = ""  # superseded by chat_model above

    # --- LLM generation params (verified against gpt-5.6 family on 2026-10-23) ---
    # reasoning_effort: "none" | "low" | "medium" | "high"
    # "none" = fastest, no internal reasoning tokens, best for chat
    chat_reasoning_effort: str = "none"
    # max_completion_tokens: total output budget (reasoning + visible tokens)
    # 500 gives ample room for 1-3 short text lines even with reasoning_effort=none
    chat_max_completion_tokens: int = 500
    # temperature: accepted by gpt-5.6 family; use 0.95 for natural variety
    chat_temperature: float = 0.95
    # top_p: accepted; set 1.0 (off) to avoid conflicting with temperature
    chat_top_p: float = 1.0

    # --- Per-user limits ---
    max_user_input_length: int = 1000          # characters; reject / truncate above this
    user_rate_limit_per_minute: int = 20       # max requests per user per minute

    # --- Database ---
    database_url: str = "sqlite:///./app.db"

    # --- ChromaDB ---
    chroma_persist_dir: str = "./chroma_data"

    # --- App behaviour ---
    cors_origins: str = ""               # Comma-separated extra origins
    max_recent_history_turns: int = 15   # raw history turns sent in prompt
    rag_top_k: int = 4                   # number of RAG chunks retrieved per query

    # Health check interval for OpenAI availability probe
    openai_health_check_interval_hours: int = 6

    # Legacy scheduler setting (kept so existing .env files don't break)
    groq_health_check_interval_hours: int = 12

    llm_request_timeout_seconds: float = 12.0  # per-call timeout (<= 15.0)
    llm_max_retries: int = 0                   # 0 = instant fallback on error

    use_direct_prompt_flow: bool = True
    enable_typing_delay: bool = True

    # --- Feature flags ---
    # Memory extraction batch: run RAG extraction every N messages (not every turn)
    memory_extraction_batch_size: int = 8
    # Daily story: generate once per calendar day per persona, reuse all day
    daily_story_enabled: bool = True
    # Proactive opener: send a context-aware message if user inactive for X seconds
    proactive_opener_idle_seconds: int = 3600   # 1 hour default
    # Streaming: enable SSE streaming to the frontend
    enable_streaming: bool = False              # set True when frontend supports it

    # --- User timezone default ---
    default_timezone: str = "Asia/Kolkata"

    # --- Cost logging ---
    # Approximate costs in USD per 1M tokens (update as pricing changes)
    cost_per_million_input_tokens: float = 0.20   # gpt-5.6-luna input
    cost_per_million_output_tokens: float = 1.20  # gpt-5.6-luna output

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    """Cached singleton — only reads .env once per process lifetime."""
    return Settings()


# Module-level convenience export
settings = get_settings()
