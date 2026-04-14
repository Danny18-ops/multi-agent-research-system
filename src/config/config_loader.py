"""Secure configuration loader backed by python-dotenv.

This module provides a single entry point — ``load_config()`` — that:

1. Locates the nearest ``.env`` file (searches upward from the cwd).
2. Loads it into the process environment without overwriting existing vars.
3. Validates every key through Pydantic and returns a typed ``AppConfig``.
4. Masks secrets when the config is printed or logged.

Usage::

    from src.config.config_loader import load_config

    cfg = load_config()
    print(cfg.openai_api_key)   # actual key
    print(cfg)                  # masked — safe to log
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel, Field, SecretStr, field_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Typed config model with secret masking
# ---------------------------------------------------------------------------


class AppConfig(BaseModel):
    """Validated, type-safe application configuration.

    Secrets (API keys) are stored as :class:`pydantic.SecretStr` so they
    are never accidentally leaked into logs or ``repr()`` output.

    Access the raw value when needed::

        cfg.openai_api_key.get_secret_value()
    """

    # ── Required ────────────────────────────────────────────────────────────
    openai_api_key: SecretStr = Field(..., description="OpenAI API key")

    # ── Model ────────────────────────────────────────────────────────────────
    model_name: str = Field("gpt-4o", description="Primary LLM model name")
    openai_temperature: float = Field(0.1, ge=0.0, le=2.0)

    # ── Search keys (all optional) ───────────────────────────────────────────
    serper_api_key: SecretStr = Field(
        default=SecretStr(""), description="Serper Google Search API key"
    )
    tavily_api_key: SecretStr = Field(
        default=SecretStr(""), description="Tavily search API key"
    )
    serpapi_api_key: SecretStr = Field(
        default=SecretStr(""), description="SerpAPI key"
    )

    # ── Agent behaviour ──────────────────────────────────────────────────────
    max_iterations: int = Field(10, ge=1, le=50)
    max_retries: int = Field(3, ge=0, le=10)
    request_timeout: int = Field(60, ge=5, le=300)
    max_search_results: int = Field(5, ge=1, le=20)

    # ── Quality ──────────────────────────────────────────────────────────────
    min_quality_score: float = Field(0.7, ge=0.0, le=1.0)

    # ── Output / logging ─────────────────────────────────────────────────────
    output_dir: str = Field("outputs")
    log_dir: str = Field("logs")
    log_level: str = Field("INFO")

    @field_validator("model_name", mode="before")
    @classmethod
    def resolve_model(cls, v: Any) -> str:
        """Accept MODEL_NAME; fall back to OPENAI_MODEL env var."""
        if v:
            return str(v)
        return os.getenv("OPENAI_MODEL", "gpt-4o")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {allowed}, got '{v}'")
        return upper

    # Pydantic v2 config
    model_config = {"extra": "ignore"}

    # ── Convenience helpers ──────────────────────────────────────────────────

    def has_serper(self) -> bool:
        """Return True when a valid Serper key is configured."""
        key = self.serper_api_key.get_secret_value()
        return bool(key and not key.startswith("your-"))

    def has_tavily(self) -> bool:
        """Return True when a valid Tavily key is configured."""
        key = self.tavily_api_key.get_secret_value()
        return bool(key and not key.startswith("your-") and not key.startswith("tvly-..."))

    def active_search_backend(self) -> str:
        """Return the name of the first configured search backend."""
        if self.has_tavily():
            return "tavily"
        if self.has_serper():
            return "serper"
        return "duckduckgo"

    def __repr__(self) -> str:  # pragma: no cover
        """Safe repr — secrets shown as '***'."""
        return (
            f"AppConfig("
            f"model_name={self.model_name!r}, "
            f"openai_api_key=***, "
            f"search_backend={self.active_search_backend()!r}, "
            f"log_level={self.log_level!r}"
            f")"
        )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_config_cache: AppConfig | None = None


def load_config(
    env_file: str | Path | None = None,
    *,
    override: bool = False,
    reload: bool = False,
) -> AppConfig:
    """Load, validate, and cache the application configuration.

    Searches for a ``.env`` file in the project tree when *env_file* is not
    given.  Existing environment variables are **not** overwritten unless
    *override* is ``True``.

    Args:
        env_file: Explicit path to a ``.env`` file.  Auto-detected when omitted.
        override: If ``True``, values in the ``.env`` file override existing
                  process environment variables.
        reload:   Force re-reading the file and bypassing the in-memory cache.

    Returns:
        A validated :class:`AppConfig` instance.

    Raises:
        pydantic.ValidationError: When a required key is missing or a value
            fails its validator.
    """
    global _config_cache  # noqa: PLW0603

    if _config_cache is not None and not reload:
        return _config_cache

    # Resolve the .env path
    if env_file is None:
        found = find_dotenv(usecwd=True)
        env_path = Path(found) if found else None
    else:
        env_path = Path(env_file)

    if env_path and env_path.exists():
        loaded = load_dotenv(dotenv_path=env_path, override=override)
        logger.debug("Loaded .env from %s (vars changed: %s)", env_path, loaded)
    else:
        logger.warning(
            "No .env file found — relying entirely on process environment variables."
        )

    # Build config from environment
    cfg = AppConfig(
        openai_api_key=SecretStr(_require("OPENAI_API_KEY")),
        model_name=os.getenv("MODEL_NAME") or os.getenv("OPENAI_MODEL", "gpt-4o"),
        openai_temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.1")),
        serper_api_key=SecretStr(os.getenv("SERPER_API_KEY", "")),
        tavily_api_key=SecretStr(os.getenv("TAVILY_API_KEY", "")),
        serpapi_api_key=SecretStr(os.getenv("SERPAPI_API_KEY", "")),
        max_iterations=int(os.getenv("MAX_ITERATIONS", "10")),
        max_retries=int(os.getenv("MAX_RETRIES", "3")),
        request_timeout=int(os.getenv("REQUEST_TIMEOUT", "60")),
        max_search_results=int(os.getenv("MAX_SEARCH_RESULTS", "5")),
        min_quality_score=float(os.getenv("MIN_QUALITY_SCORE", "0.7")),
        output_dir=os.getenv("OUTPUT_DIR", "outputs"),
        log_dir=os.getenv("LOG_DIR", "logs"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )

    _config_cache = cfg
    logger.info("Config loaded — %r", cfg)
    return cfg


def _require(key: str) -> str:
    """Return env var *key* or raise a clear error if it is missing/empty."""
    value = os.getenv(key, "").strip()
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            f"Add it to your .env file or export it before running."
        )
    return value
