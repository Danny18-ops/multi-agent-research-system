"""Web scraping tool — fetches and cleans page content for agents."""

from __future__ import annotations

import logging
import re
from urllib.parse import urlparse

import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; ResearchBot/1.0; "
        "+https://github.com/example/multi-agent-research)"
    )
}


class ScrapeInput(BaseModel):
    """Input schema for the web scraper tool."""

    url: str = Field(..., description="The full URL of the page to scrape")
    max_chars: int = Field(
        4000, ge=100, le=20_000, description="Maximum characters to return"
    )


class WebScraperTool(BaseTool):
    """Fetch and extract readable text from a web page.

    The tool strips HTML tags, scripts, and style blocks, returning
    only the meaningful body text — ready for an LLM to process.
    """

    name: str = "web_scraper"
    description: str = (
        "Fetch and extract clean text content from a webpage URL. "
        "Use this after finding relevant URLs via search to get the full article content. "
        "Returns readable text stripped of HTML tags."
    )
    args_schema: type[BaseModel] = ScrapeInput

    request_timeout: int = 30

    def _run(self, url: str, max_chars: int = 4000) -> str:
        """Fetch *url* and return cleaned plain text up to *max_chars*."""
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return f"Invalid URL scheme '{parsed.scheme}'. Only http/https is supported."

        try:
            response = requests.get(
                url, headers=_HEADERS, timeout=self.request_timeout
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("Failed to fetch %s: %s", url, exc)
            return f"Could not fetch page: {exc}"

        content_type = response.headers.get("content-type", "")
        if "text/html" not in content_type and "text/plain" not in content_type:
            return f"Unsupported content type: {content_type}"

        text = self._clean_html(response.text)

        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n[Content truncated…]"

        logger.info("Scraped %d chars from %s", len(text), url)
        return text

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_html(raw_html: str) -> str:
        """Strip HTML tags and normalise whitespace."""
        # Remove script and style blocks wholesale
        cleaned = re.sub(
            r"<(script|style|noscript)[^>]*>.*?</\1>",
            " ",
            raw_html,
            flags=re.DOTALL | re.IGNORECASE,
        )
        # Remove all remaining HTML tags
        cleaned = re.sub(r"<[^>]+>", " ", cleaned)
        # Decode common HTML entities
        entities = {
            "&amp;": "&",
            "&lt;": "<",
            "&gt;": ">",
            "&quot;": '"',
            "&#39;": "'",
            "&nbsp;": " ",
        }
        for entity, char in entities.items():
            cleaned = cleaned.replace(entity, char)
        # Collapse whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned
