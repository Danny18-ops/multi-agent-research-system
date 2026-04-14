"""Web search tools for the Research Agent.

Three backends are provided, in order of preference:

1. **SerperSearchTool** — Google results via Serper API (recommended).
2. **TavilySearchTool** — LLM-optimised results via Tavily AI.
3. **WebSearchTool**    — Free DuckDuckGo fallback; no API key required.

All tools share a common retry strategy (exponential back-off via
``tenacity``) and return a uniform plain-text format that the agent LLM
can parse consistently.
"""

from __future__ import annotations

import logging
from typing import Any

import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from tenacity import (
    RetryError,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.models.research_models import SearchBackend, SourceResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared input schema
# ---------------------------------------------------------------------------


class SearchInput(BaseModel):
    """Input schema shared by all search tools."""

    query: str = Field(..., description="The search query string")
    max_results: int = Field(5, ge=1, le=20, description="Maximum number of results")


# ---------------------------------------------------------------------------
# Shared retry decorator factory
# ---------------------------------------------------------------------------


def _make_retry(max_attempts: int = 3):
    """Return a tenacity ``@retry`` decorator with exponential back-off.

    Retries on any :class:`requests.RequestException` (connection errors,
    timeouts, HTTP 5xx responses raised via ``raise_for_status``).
    """
    return retry(
        retry=retry_if_exception_type(requests.RequestException),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=False,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _relevance_from_position(position: int, max_results: int) -> float:
    """Convert a 1-based search rank to a 0–1 relevance score.

    Position 1 → 1.0; last position → ~0.1.
    """
    if max_results <= 1:
        return 1.0
    score = 1.0 - (position - 1) / max_results
    return round(max(0.1, min(1.0, score)), 3)


def _format_sources(sources: list[SourceResult]) -> str:
    """Render a list of :class:`SourceResult` objects as agent-readable text."""
    if not sources:
        return "No results found."
    lines: list[str] = []
    for src in sources:
        lines.append(
            f"[{src.position}] {src.title}\n"
            f"    URL: {src.url}\n"
            f"    Relevance: {src.relevance_score:.2f}\n"
            f"    Snippet: {src.snippet[:400]}\n"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Serper (Google Search API)
# ---------------------------------------------------------------------------


class SerperSearchTool(BaseTool):
    """Search Google via the Serper API with automatic retry on failure.

    Serper returns real Google results (organic, news, etc.) in a clean
    JSON format.  Get a free API key at https://serper.dev.

    Retry behaviour:
    - Up to ``max_retries`` attempts on any network or HTTP error.
    - Exponential back-off: 2s → 4s → 8s … (capped at 15s).
    - Returns an error string (never raises) so the agent can decide how
      to handle the failure.

    Example::

        tool = SerperSearchTool(api_key="abc123", max_retries=3)
        text = tool._run("quantum computing breakthroughs 2024", max_results=5)
    """

    name: str = "serper_web_search"
    description: str = (
        "Search Google using the Serper API. "
        "Returns ranked results with titles, URLs, and snippets. "
        "Best for finding recent, authoritative information on any topic. "
        "Input: a natural-language search query."
    )
    args_schema: type[BaseModel] = SearchInput

    api_key: str = ""
    max_retries: int = 3
    request_timeout: int = 30
    base_url: str = "https://google.serper.dev/search"

    def _run(self, query: str, max_results: int = 5) -> str:
        """Execute a Serper search and return formatted results.

        Args:
            query:       Natural-language search query.
            max_results: Maximum number of organic results to return.

        Returns:
            Formatted string of results, or an error message.
        """
        if not self.api_key:
            logger.warning("SerperSearchTool: no API key configured")
            return "Serper API key not configured. Please set SERPER_API_KEY."

        logger.info("SerperSearchTool: querying %r (max=%d)", query, max_results)

        try:
            raw = self._fetch_with_retry(query, max_results)
        except RetryError as exc:
            msg = f"Serper search failed after {self.max_retries} attempts: {exc}"
            logger.error(msg)
            return msg

        sources = self._parse_response(raw, max_results)
        logger.info("SerperSearchTool: returned %d results", len(sources))
        return _format_sources(sources)

    def _fetch_with_retry(self, query: str, max_results: int) -> dict[str, Any]:
        """HTTP call wrapped in tenacity retry logic."""

        @_make_retry(self.max_retries)
        def _call() -> dict[str, Any]:
            resp = requests.post(
                self.base_url,
                headers={
                    "X-API-KEY": self.api_key,
                    "Content-Type": "application/json",
                },
                json={"q": query, "num": max_results},
                timeout=self.request_timeout,
            )
            resp.raise_for_status()
            return resp.json()  # type: ignore[no-any-return]

        return _call()

    def _parse_response(
        self, data: dict[str, Any], max_results: int
    ) -> list[SourceResult]:
        """Convert a raw Serper API response to ``SourceResult`` objects."""
        sources: list[SourceResult] = []

        organic: list[dict] = data.get("organic", [])
        for item in organic[:max_results]:
            pos = item.get("position", len(sources) + 1)
            sources.append(
                SourceResult(
                    url=item.get("link", ""),
                    title=item.get("title", ""),
                    snippet=item.get("snippet", ""),
                    relevance_score=_relevance_from_position(pos, max_results),
                    position=pos,
                    backend=SearchBackend.SERPER,
                )
            )

        return sources

    def search_structured(
        self, query: str, max_results: int = 5
    ) -> list[SourceResult]:
        """Return raw :class:`SourceResult` objects instead of formatted text.

        Useful for programmatic callers (e.g. ``ResearchRunner``) that need
        the structured data rather than the string representation.
        """
        if not self.api_key:
            return []
        try:
            raw = self._fetch_with_retry(query, max_results)
            return self._parse_response(raw, max_results)
        except (RetryError, Exception) as exc:  # noqa: BLE001
            logger.error("SerperSearchTool.search_structured failed: %s", exc)
            return []


# ---------------------------------------------------------------------------
# Tavily (LLM-optimised search)
# ---------------------------------------------------------------------------


class TavilySearchTool(BaseTool):
    """Search the web via the Tavily AI search API with retry.

    Tavily is optimised for LLM consumption: it pre-processes results and
    provides a concise AI-generated answer alongside raw snippets.
    Get a key at https://tavily.com.
    """

    name: str = "tavily_web_search"
    description: str = (
        "Search the internet using Tavily AI. "
        "Returns structured results with titles, URLs, and content snippets. "
        "Also provides an AI-generated summary answer. "
        "Best for factual, in-depth research on any topic."
    )
    args_schema: type[BaseModel] = SearchInput

    api_key: str = ""
    max_retries: int = 3
    request_timeout: int = 30
    base_url: str = "https://api.tavily.com/search"

    def _run(self, query: str, max_results: int = 5) -> str:
        """Execute a Tavily search and return formatted results."""
        if not self.api_key:
            logger.warning("TavilySearchTool: no API key configured")
            return "Tavily API key not configured. Please set TAVILY_API_KEY."

        logger.info("TavilySearchTool: querying %r (max=%d)", query, max_results)

        try:
            raw = self._fetch_with_retry(query, max_results)
        except RetryError as exc:
            msg = f"Tavily search failed after {self.max_retries} attempts: {exc}"
            logger.error(msg)
            return msg

        sources = self._parse_response(raw, max_results)
        output_parts: list[str] = []

        if answer := raw.get("answer"):
            output_parts.append(f"AI Summary: {answer}\n")

        output_parts.append(_format_sources(sources))
        logger.info("TavilySearchTool: returned %d results", len(sources))
        return "\n".join(output_parts)

    def _fetch_with_retry(self, query: str, max_results: int) -> dict[str, Any]:
        """HTTP call wrapped in tenacity retry logic."""

        @_make_retry(self.max_retries)
        def _call() -> dict[str, Any]:
            resp = requests.post(
                self.base_url,
                json={
                    "api_key": self.api_key,
                    "query": query,
                    "search_depth": "advanced",
                    "max_results": max_results,
                    "include_answer": True,
                    "include_raw_content": False,
                },
                timeout=self.request_timeout,
            )
            resp.raise_for_status()
            return resp.json()  # type: ignore[no-any-return]

        return _call()

    def _parse_response(
        self, data: dict[str, Any], max_results: int
    ) -> list[SourceResult]:
        """Convert a raw Tavily API response to ``SourceResult`` objects."""
        sources: list[SourceResult] = []
        for pos, item in enumerate(data.get("results", [])[:max_results], start=1):
            raw_score = item.get("score", _relevance_from_position(pos, max_results))
            sources.append(
                SourceResult(
                    url=item.get("url", ""),
                    title=item.get("title", ""),
                    snippet=item.get("content", ""),
                    relevance_score=raw_score,
                    position=pos,
                    backend=SearchBackend.TAVILY,
                )
            )
        return sources

    def search_structured(
        self, query: str, max_results: int = 5
    ) -> list[SourceResult]:
        """Return :class:`SourceResult` objects instead of formatted text."""
        if not self.api_key:
            return []
        try:
            raw = self._fetch_with_retry(query, max_results)
            return self._parse_response(raw, max_results)
        except (RetryError, Exception) as exc:  # noqa: BLE001
            logger.error("TavilySearchTool.search_structured failed: %s", exc)
            return []


# ---------------------------------------------------------------------------
# DuckDuckGo (free fallback — no API key)
# ---------------------------------------------------------------------------


class WebSearchTool(BaseTool):
    """Search the web using DuckDuckGo — no API key required.

    This is the default fallback when no commercial API key is configured.
    Results are less comprehensive than Serper or Tavily but require zero
    configuration.
    """

    name: str = "web_search"
    description: str = (
        "Search the internet using DuckDuckGo (no API key required). "
        "Returns search result titles, URLs, and snippets. "
        "Use this as a fallback when other search backends are unavailable."
    )
    args_schema: type[BaseModel] = SearchInput

    max_retries: int = 2

    def _run(self, query: str, max_results: int = 5) -> str:
        """Execute a DuckDuckGo search and return formatted results."""
        try:
            from duckduckgo_search import DDGS  # type: ignore[import-untyped]
        except ImportError:
            return (
                "duckduckgo-search package not installed. "
                "Run: pip install duckduckgo-search"
            )

        logger.info("WebSearchTool (DDG): querying %r (max=%d)", query, max_results)

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                with DDGS() as ddgs:
                    raw = list(ddgs.text(query, max_results=max_results))
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning(
                    "WebSearchTool attempt %d/%d failed: %s",
                    attempt,
                    self.max_retries,
                    exc,
                )
        else:
            return f"DuckDuckGo search failed after {self.max_retries} attempts: {last_exc}"

        if not raw:
            return "No results found."

        sources = [
            SourceResult(
                url=r.get("href", ""),
                title=r.get("title", ""),
                snippet=r.get("body", ""),
                relevance_score=_relevance_from_position(i, max_results),
                position=i,
                backend=SearchBackend.DUCKDUCKGO,
            )
            for i, r in enumerate(raw, start=1)
            if r.get("href")
        ]

        logger.info("WebSearchTool: returned %d results", len(sources))
        return _format_sources(sources)
