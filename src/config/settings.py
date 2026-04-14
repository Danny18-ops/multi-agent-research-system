"""Application settings loaded from environment variables."""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    """Central configuration for the Multi-Agent Research System.

    All values are read from environment variables (or a .env file).
    Call ``get_settings()`` to obtain the cached singleton.
    """

    # --- LLM ---
    openai_api_key: str = Field(..., description="OpenAI API key")
    # MODEL_NAME is the canonical setting; OPENAI_MODEL is the legacy alias.
    # If both are set, MODEL_NAME wins (resolved in model_validator below).
    model_name: str = Field("gpt-4o", description="Primary model name for all agents")
    openai_model: str = Field("gpt-4o", description="Legacy alias for model_name")
    openai_temperature: float = Field(0.1, ge=0.0, le=2.0)

    # --- Search ---
    serper_api_key: str = Field("", description="Serper Google Search API key")
    tavily_api_key: str = Field("", description="Tavily search API key (optional)")
    serpapi_api_key: str = Field("", description="SerpAPI key (optional fallback)")
    max_search_results: int = Field(5, ge=1, le=20)

    # --- Agent behaviour ---
    max_iterations: int = Field(10, ge=1, le=50, description="Max agent reasoning steps")
    max_retries: int = Field(3, ge=0, le=10)
    request_timeout: int = Field(60, ge=5, le=300, description="HTTP timeout in seconds")

    # --- Output ---
    output_dir: str = Field("outputs", description="Directory for generated reports")
    log_dir: str = Field("logs", description="Directory for log files")
    log_level: str = Field("INFO", description="Logging verbosity")

    # --- Quality check thresholds ---
    min_quality_score: float = Field(
        0.7, ge=0.0, le=1.0, description="Minimum report quality score (0–1)"
    )

    @model_validator(mode="after")
    def resolve_model_name(self) -> "Settings":
        """Make MODEL_NAME authoritative; keep openai_model in sync."""
        # If MODEL_NAME was explicitly set in env, honour it over OPENAI_MODEL
        if self.model_name and self.model_name != "gpt-4o":
            self.openai_model = self.model_name
        elif self.openai_model and self.openai_model != "gpt-4o":
            self.model_name = self.openai_model
        return self

    @property
    def effective_model(self) -> str:
        """Return the resolved model name (MODEL_NAME > OPENAI_MODEL > default)."""
        return self.model_name or self.openai_model or "gpt-4o"

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return upper

    @field_validator("output_dir", "log_dir")
    @classmethod
    def ensure_directory_exists(cls, v: str) -> str:
        os.makedirs(v, exist_ok=True)
        return v

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings singleton."""
    return Settings()  # type: ignore[call-arg]
