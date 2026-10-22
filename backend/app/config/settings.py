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
    groq_api_key: str = ""
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    # --- Model Names ---
    groq_model: str = "llama-3.3-70b-versatile"
    claude_model: str = "claude-3-5-haiku-20241022"
    openai_model: str = "gpt-4o-mini"

    # --- Database ---
    database_url: str = "sqlite:///./app.db"

    # --- ChromaDB ---
    chroma_persist_dir: str = "./chroma_data"

    # --- App behaviour ---
    cors_origins: str = ""               # Comma-separated extra origins (e.g. https://myfrontend.com)
    max_recent_history_turns: int = 10   # raw history turns sent in prompt
    rag_top_k: int = 4                   # number of RAG chunks retrieved per query
    groq_health_check_interval_hours: int = 12
    llm_request_timeout_seconds: float = 12.0  # Fast failover ceiling per provider (never stalls chat)
    llm_max_retries: int = 0                   # 0 retries allows instant failover to Claude on rate limits
    use_direct_prompt_flow: bool = True        # Direct OpenAI query without restrictive rules; format via Python
    enable_typing_delay: bool = True           # Realistic human typing delay based on message length

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
