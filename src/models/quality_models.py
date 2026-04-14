"""Data models for the Quality Checker Agent pipeline.

All models are Pydantic v2 and fully JSON-serialisable so they can be
persisted, logged, or passed between workflow steps without loss.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class IssueSeverity(str, Enum):
    """How severely an issue affects the report's overall quality."""

    CRITICAL = "critical"  # factual error, broken citation — must fix
    MAJOR = "major"        # significant coverage gap or logical flaw
    MINOR = "minor"        # style, wording, or minor omission


class DimensionName(str, Enum):
    """The four quality dimensions evaluated by the agent."""

    FACTUAL_CONSISTENCY = "factual_consistency"
    CITATION_ACCURACY = "citation_accuracy"
    COMPLETENESS = "completeness"
    LOGICAL_FLOW = "logical_flow"


# ---------------------------------------------------------------------------
# Core sub-models
# ---------------------------------------------------------------------------


class QualityDimension(BaseModel):
    """Score and feedback for one of the four evaluation dimensions.

    Attributes:
        name:         Machine-readable dimension identifier.
        display_name: Human-readable label (e.g. ``"Factual Consistency"``).
        score:        Integer score 0–25 awarded by the agent.
        max_score:    Upper bound for this dimension (always 25).
        issues:       Specific problems found in this dimension.
        suggestions:  Actionable fixes for the issues found.
    """

    name: str = Field(..., description="Machine-readable dimension name")
    display_name: str = Field(..., description="Human-readable dimension label")
    score: int = Field(0, ge=0, le=25, description="Score awarded (0–25)")
    max_score: int = Field(25, ge=1)
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)

    @field_validator("score", mode="before")
    @classmethod
    def clamp_score(cls, v: Any) -> int:
        try:
            return max(0, min(25, int(v)))
        except (TypeError, ValueError):
            return 0

    @property
    def percentage(self) -> float:
        """Score as a percentage of the max (0–100)."""
        return round((self.score / self.max_score) * 100, 1) if self.max_score > 0 else 0.0

    @property
    def passed(self) -> bool:
        """``True`` when this dimension scores ≥ 70 %."""
        return self.percentage >= 70.0


class QualityIssue(BaseModel):
    """A single specific problem found during quality checking.

    Attributes:
        dimension:   Which of the four checks discovered the issue.
        severity:    Critical / major / minor.
        description: Clear description of what is wrong.
        location:    Optional pointer to where in the report the issue appears.
        suggestion:  Concrete step the report writer should take to fix it.
    """

    dimension: str = Field(..., description="Dimension that discovered this issue")
    severity: IssueSeverity = Field(IssueSeverity.MINOR)
    description: str = Field(..., description="What is wrong")
    location: str | None = Field(None, description="Where in the report")
    suggestion: str = Field("", description="How to fix it")

    @field_validator("severity", mode="before")
    @classmethod
    def coerce_severity(cls, v: Any) -> IssueSeverity:
        if isinstance(v, IssueSeverity):
            return v
        try:
            return IssueSeverity(str(v).lower())
        except ValueError:
            return IssueSeverity.MINOR


# ---------------------------------------------------------------------------
# Main quality report
# ---------------------------------------------------------------------------


class QualityReport(BaseModel):
    """Complete output of one quality-check run.

    The ``overall_score`` is the sum of the four dimension scores (max 100).
    When it falls below the configured threshold (default 70), ``revision_required``
    is ``True`` and ``revision_prompt`` carries targeted instructions for the
    Report Writer Agent.

    Attributes:
        topic:             Research topic the report covers.
        overall_score:     Aggregate quality score 0–100.
        dimensions:        Per-dimension breakdown (exactly four entries).
        issues:            Flat list of all :class:`QualityIssue` objects found.
        top_suggestions:   Up to five prioritised improvement actions.
        verdict:           ``"APPROVED"`` or ``"REVISION_REQUIRED"``.
        revision_required: ``True`` when ``overall_score < quality_threshold``.
        revision_prompt:   Targeted revision instructions for the report writer,
                           or ``None`` when approved.
        success:           ``True`` when the check completed without error.
        error:             Error message if ``success`` is ``False``.
        duration_secs:     Wall-clock time for the check.
        retry_count:       How many crew-level attempts were needed.
    """

    topic: str = Field(default="", description="Research topic")
    overall_score: int = Field(0, ge=0, le=100)
    dimensions: list[QualityDimension] = Field(default_factory=list)
    issues: list[QualityIssue] = Field(default_factory=list)
    top_suggestions: list[str] = Field(default_factory=list)
    verdict: Literal["APPROVED", "REVISION_REQUIRED"] = "REVISION_REQUIRED"
    revision_required: bool = True
    revision_prompt: str | None = None
    success: bool = False
    error: str | None = None
    duration_secs: float = Field(0.0, ge=0.0)
    retry_count: int = Field(0, ge=0)

    @field_validator("overall_score", mode="before")
    @classmethod
    def clamp_overall(cls, v: Any) -> int:
        try:
            return max(0, min(100, int(v)))
        except (TypeError, ValueError):
            return 0

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def has_critical_issues(self) -> bool:
        """``True`` when at least one CRITICAL issue was found."""
        return any(i.severity == IssueSeverity.CRITICAL for i in self.issues)

    @property
    def dimension_map(self) -> dict[str, QualityDimension]:
        """Dimension objects keyed by their ``name`` for O(1) lookup."""
        return {d.name: d for d in self.dimensions}

    @property
    def failed_dimensions(self) -> list[QualityDimension]:
        """Dimensions that scored below 70 %."""
        return [d for d in self.dimensions if not d.passed]

    def get_dimension(self, name: str) -> QualityDimension | None:
        """Return the named dimension or ``None``."""
        return self.dimension_map.get(name)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def to_markdown(self) -> str:
        """Render the full quality report as a Markdown string."""
        verdict_icon = "✅" if self.verdict == "APPROVED" else "⚠️"
        lines: list[str] = [
            f"# Quality Check Report",
            f"**Topic:** {self.topic}",
            "",
            f"## Overall Score: {self.overall_score} / 100 — {verdict_icon} {self.verdict}",
            "",
            "## Dimension Scores",
            "",
            "| Dimension | Score | / Max | % | Status |",
            "|---|---:|---:|---:|:---|",
        ]
        for dim in self.dimensions:
            status = "✅ Pass" if dim.passed else "❌ Fail"
            lines.append(
                f"| {dim.display_name} | {dim.score} | {dim.max_score}"
                f" | {dim.percentage:.0f}% | {status} |"
            )

        if self.issues:
            lines += ["", "## Issues Found", ""]
            for issue in sorted(self.issues, key=lambda x: x.severity.value):
                loc = f" *(in: {issue.location})*" if issue.location else ""
                lines.append(
                    f"- **[{issue.severity.value.upper()}]**{loc} "
                    f"{issue.description}"
                )
                if issue.suggestion:
                    lines.append(f"  - *Fix:* {issue.suggestion}")

        if self.top_suggestions:
            lines += ["", "## Top Improvement Suggestions", ""]
            for i, sug in enumerate(self.top_suggestions, 1):
                lines.append(f"{i}. {sug}")

        if self.revision_required and self.revision_prompt:
            lines += ["", "## Revision Instructions for Report Writer", "", self.revision_prompt]

        if not self.success and self.error:
            lines += ["", f"> **Error:** {self.error}"]

        lines += ["", f"*Check completed in {self.duration_secs:.1f}s — attempt(s): {self.retry_count + 1}*"]
        return "\n".join(lines)
