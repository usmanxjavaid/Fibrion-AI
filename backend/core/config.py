"""
Centralized application settings.

Every other module imports `settings` from here instead of calling
os.getenv() directly. Required values are validated the moment this
module is imported — if something's missing, the app fails immediately
and loudly at startup, not three agents deep into a pipeline run.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM API ---
    openrouter_api_key: str
    # default_model_fast: str = "anthropic/claude-haiku-4-5"
    default_model_fast: str = "nvidia/nemotron-3-super-120b-a12b:free"
    default_model_reasoning: str = "anthropic/claude-sonnet-5"
    groq_api_key: str = ""
    gemini_api_key: str = ""
    cerebras_api_key: str = ""
    

    # --- Telegram ---
    telegram_bot_token: str

    # --- Email ---
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587       # STARTTLS — Gmail's standard secure SMTP port
    email_address: str = ""
    email_app_password: str = ""

    # --- App ---
    fibrion_env: str = "development"
    log_level: str = "INFO"

    # --- Pipeline behavior ---
    verification_max_retries: int = 1

    # --- Observability (optional - tracing only activates if enabled) ---
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "fibrion"


settings = Settings()

# LangChain's own tracing reads these as real OS environment variables,
# not from our settings object - this is the one place that translation
# happens, so every other file can just import `settings` normally.
if settings.langsmith_tracing:
    import os
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
