import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base directory paths
APP_DIR = Path(__file__).resolve().parent
ECOM_ROOT = APP_DIR.parent
DATA_DIR = ECOM_ROOT / "data"
POLICIES_DIR = DATA_DIR / "policies"
STATIC_DIR = APP_DIR / "static"

# Pre-populate os.environ from .env so third-party SDKs (Langfuse, OpenAI) read them at module load time
env_candidates = [
    ECOM_ROOT / ".env",
    ECOM_ROOT.parent / ".env",
]
for p in env_candidates:
    if p.exists():
        load_dotenv(p, override=False)
        break


class Settings(BaseSettings):
    # Database (loaded from .env)
    DATABASE_URL: str = Field(
        ...,
        description="Async database connection string (e.g. postgresql+asyncpg://... or sqlite+aiosqlite:///...)",
    )

    # JWT Authentication (secret loaded strictly from .env)
    JWT_SECRET: str = Field(
        ...,
        description="Secret key for signing and verifying JWT tokens",
    )
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 480

    # LLM Settings (API key loaded from .env)
    LLM_PROVIDER: str = "groq"  # "groq" or "ollama"
    LLM_MODEL: str = "openai/gpt-oss-120b"
    LLM_BASE_URL: str = "https://api.groq.com/openai/v1"
    LLM_API_KEY: str = Field(
        ...,
        description="API key for LLM provider (Groq API key or 'ollama')",
    )
    GROQ_API_KEY: Optional[str] = Field(
        default=None,
        description="Explicit Groq API key (falls back to LLM_API_KEY if not specified)",
    )
    LLM_TIMEOUT: float = Field(15.0, description="HTTP timeout (seconds) for external LLM inference")

    # Intent-Safety Classifier / Guardrails Settings
    GUARDRAIL_ENABLED: bool = Field(True, description="Enable multi-layer intent and prompt injection guardrails")
    GUARDRAIL_INTENT_MODEL: str = Field(
        "openai/gpt-oss-20b",
        description="LLM classifier model for business domain and prompt injection classification",
    )
    GUARDRAIL_PROMPT_GUARD_MODEL: str = Field(
        "meta-llama/llama-prompt-guard-2-86m",
        description="Dedicated Prompt Guard model on Groq for injection scoring",
    )
    GUARDRAIL_PROMPT_GUARD_THRESHOLD: float = Field(
        0.85,
        description="Threshold probability for Meta Prompt Guard injection detection",
    )
    GUARDRAIL_TIMEOUT: float = Field(
        5.0,
        description="HTTP timeout (seconds) for intent-safety guardrail inference",
    )

    # Embedding Model
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384

    # System parameters
    MAX_ITERATIONS: int = 5
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # Langfuse Observability
    LANGFUSE_PUBLIC_KEY: Optional[str] = None
    LANGFUSE_SECRET_KEY: Optional[str] = None
    LANGFUSE_BASE_URL: Optional[str] = None
    LANGFUSE_HOST: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=(str(ECOM_ROOT / ".env"), str(ECOM_ROOT.parent / ".env")),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def is_postgres(self) -> bool:
        return "postgresql" in self.DATABASE_URL.lower()

    @property
    def is_sqlite(self) -> bool:
        return "sqlite" in self.DATABASE_URL.lower()

    @property
    def langfuse_host(self) -> str:
        raw = self.LANGFUSE_BASE_URL or self.LANGFUSE_HOST or os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST") or "https://cloud.langfuse.com"
        return raw.strip("\"'")

    @property
    def groq_api_key(self) -> str:
        raw = self.GROQ_API_KEY or self.LLM_API_KEY or os.getenv("GROQ_API_KEY") or os.getenv("LLM_API_KEY") or ""
        return raw.strip("\"'")


settings = Settings()

# Ensure Langfuse SDK environment variables are cleanly synchronized to os.environ
if settings.LANGFUSE_PUBLIC_KEY:
    clean_pk = settings.LANGFUSE_PUBLIC_KEY.strip("\"'")
    os.environ["LANGFUSE_PUBLIC_KEY"] = clean_pk

if settings.LANGFUSE_SECRET_KEY:
    clean_sk = settings.LANGFUSE_SECRET_KEY.strip("\"'")
    os.environ["LANGFUSE_SECRET_KEY"] = clean_sk

clean_host = settings.langfuse_host
os.environ["LANGFUSE_BASE_URL"] = clean_host
os.environ["LANGFUSE_HOST"] = clean_host
