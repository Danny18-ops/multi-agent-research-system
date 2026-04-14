"""Structured output models for the Research Agent pipeline.

All models are Pydantic v2 and are fully serialisable to/from JSON so
they can be persisted, shipped over an API, or passed between agents.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class SearchBackend(str, Enum):
    """Which search API was used to gather sources."""

    SERPER = "serper"
    TAVILY = "tavily"
    DUCKDUCKGO = "duckduckgo"
    UNKNOWN = "unknown"


class SourceResult(BaseModel):
    """A single web source retrieved and analysed during research.

    Attributes:
        url:             The canonical URL of the source page.
        title:           Page or article title.
        snippet:         Raw excerpt returned by the search engine.
        key_points:      Agent-extracted bullet insights from this source.
        relevance_score: Float 0.0–1.0 indicating how relevant this source
                         is to the research topic.
        position:        Rank position in the original search results (1-based).
        backend:         Which search API produced this result.
    """

    url: str = Field(..., description="Canonical URL of the source")
    title: str = Field(default="", description="Page or article title")
    snippet: str = Field(default="", description="Raw search-engine excerpt")
    key_points: list[str] = Field(
        default_factory=list,
        description="Agent-extracted key insights from this source",
    )
    relevance_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Relevance to the research topic (0–1)",
    )
    position: int = Field(
        default=0,
        ge=0,
        description="Rank position in original search results (1-based; 0 = unknown)",
    )
    backend: SearchBackend = Field(
        default=SearchBackend.UNKNOWN,
        description="Search backend that returned this result",
    )

    @field_validator("url")
    @classmethod
    def url_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("url must not be empty")
        return v.strip()

    @field_validator("relevance_score", mode="before")
    @classmethod
    def clamp_score(cls, v: Any) -> float:
        """Accept int/float and clamp to [0.0, 1.0]."""
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, f))

    def to_markdown(self) -> str:
        """Render the source as a Markdown block."""
        lines = [
            f"### [{self.title or self.url}]({self.url})",
            f"- **Relevance:** {self.relevance_score:.2f}",
            f"- **Backend:** {self.backend.value}",
        ]
        if self.snippet:
            lines.append(f"- **Snippet:** {self.snippet[:200]}")
        if self.key_points:
            lines.append("- **Key points:**")
            lines.extend(f"  - {pt}" for pt in self.key_points)
        return "\n".join(lines)


class ResearchOutput(BaseModel):
    """Complete, structured output of one Research Agent run.

    Attributes:
        topic:                   The original research topic string.
        sources:                 All sources retrieved and scored.
        key_findings:            Synthesised cross-source insights.
        search_queries:          All queries sent to the search backend.
        total_sources_examined:  Total URLs considered before deduplication.
        success:                 ``True`` when the run completed without error.
        error:                   Error message if ``success`` is ``False``.
        duration_secs:           Wall-clock runtime for the full run.
        backend:                 Search backend used.
        retry_count:             How many attempts were made before success.
    """

    topic: str = Field(..., description="The original research topic")
    sources: list[SourceResult] = Field(
        default_factory=list,
        description="Sources retrieved and scored",
    )
    key_findings: list[str] = Field(
        default_factory=list,
        description="Synthesised cross-source insights",
    )
    search_queries: list[str] = Field(
        default_factory=list,
        description="All queries submitted to the search backend",
    )
    total_sources_examined: int = Field(
        default=0,
        ge=0,
        description="Total URLs considered before deduplication",
    )
    success: bool = Field(default=False)
    error: str | None = Field(default=None)
    duration_secs: float = Field(default=0.0, ge=0.0)
    backend: SearchBackend = Field(default=SearchBackend.UNKNOWN)
    retry_count: int = Field(default=0, ge=0)

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def top_sources(self) -> list[SourceResult]:
        """Return sources sorted by relevance descending."""
        return sorted(self.sources, key=lambda s: s.relevance_score, reverse=True)

    @property
    def avg_relevance(self) -> float:
        """Mean relevance score across all sources (0.0 if none)."""
        if not self.sources:
            return 0.0
        return sum(s.relevance_score for s in self.sources) / len(self.sources)

    def to_markdown(self) -> str:
        """Render the full research output as a Markdown report."""
        lines = [
            f"# Research Report: {self.topic}",
            "",
            f"**Backend:** {self.backend.value}  |  "
            f"**Sources:** {len(self.sources)}  |  "
            f"**Avg relevance:** {self.avg_relevance:.2f}  |  "
            f"**Duration:** {self.duration_secs:.1f}s",
            "",
            "## Key Findings",
        ]
        for i, finding in enumerate(self.key_findings, 1):
            lines.append(f"{i}. {finding}")

        lines += ["", "## Sources"]
        for src in self.top_sources:
            lines += ["", src.to_markdown()]

        if not self.success and self.error:
            lines += ["", f"> **Error:** {self.error}"]

        return "\n".join(lines)
