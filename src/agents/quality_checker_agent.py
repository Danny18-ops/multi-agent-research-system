"""Quality Checker Agent and QualityCheckerRunner.

``QualityCheckerAgent`` is a CrewAI Agent factory.
``QualityCheckerRunner`` is the high-level orchestrator that:

- Evaluates a report across four scored dimensions (total 0–100):
    1. Factual Consistency  (0–25) — claims match research evidence
    2. Citation Accuracy    (0–25) — sources cited correctly and plausibly
    3. Completeness         (0–25) — all key research findings covered
    4. Logical Flow         (0–25) — structure, transitions, coherent narrative

- Returns a fully typed :class:`~src.models.QualityReport`
- When ``overall_score < quality_threshold`` (default 70), sets
  ``revision_required = True`` and populates ``revision_prompt``
- ``run_with_revision_loop`` wires the checker to any revision callable,
  retrying until the score meets the threshold or ``max_rounds`` is exhausted

Typical usage::

    from src.agents.quality_checker_agent import QualityCheckerRunner

    checker = QualityCheckerRunner(quality_threshold=70)

    # Single check
    report = checker.check(report_content="...", topic="AI in healthcare")

    # Revision loop
    final_report, final_quality = checker.run_with_revision_loop(
        report_content=initial_md,
        revise_fn=lambda report, notes: report_writer_runner.revise(report, notes),
        topic="AI in healthcare",
        max_rounds=3,
    )
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import TYPE_CHECKING, Any, Callable

from crewai import Agent, Crew, Process, Task

from src.models.quality_models import (
    DimensionName,
    IssueSeverity,
    QualityDimension,
    QualityIssue,
    QualityReport,
)

if TYPE_CHECKING:
    from crewai.tools import BaseTool
    from src.models.research_models import ResearchOutput

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_QUALITY_JSON_START = "QUALITY_JSON_START"
_QUALITY_JSON_END = "QUALITY_JSON_END"

_DEFAULT_THRESHOLD = 70

# Four fixed dimensions with their display names and max scores
_DIMENSION_META: list[dict[str, Any]] = [
    {"name": DimensionName.FACTUAL_CONSISTENCY, "display": "Factual Consistency"},
    {"name": DimensionName.CITATION_ACCURACY,   "display": "Citation Accuracy"},
    {"name": DimensionName.COMPLETENESS,         "display": "Completeness of Coverage"},
    {"name": DimensionName.LOGICAL_FLOW,         "display": "Logical Flow"},
]


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


class QualityCheckerAgent:
    """Factory for a CrewAI Quality Checker Agent.

    The agent is responsible for:
    - Comparing the report against the original research evidence
    - Scoring each of the four quality dimensions (0–25 each)
    - Listing specific issues with severity labels
    - Generating actionable improvement suggestions
    - Producing a structured JSON block parseable by :class:`QualityCheckerRunner`

    Example::

        agent = QualityCheckerAgent.create(tools=[], model="gpt-4o")
    """

    @staticmethod
    def create(
        tools: list[BaseTool] | None = None,
        model: str = "gpt-4o",
        verbose: bool = True,
        max_iter: int = 8,
    ) -> Agent:
        """Instantiate and return a configured CrewAI ``Agent``.

        Args:
            tools:    Optional CrewAI tools (e.g. search for fact-checking).
            model:    OpenAI model identifier.
            verbose:  Log chain-of-thought.
            max_iter: Maximum reasoning iterations.

        Returns:
            A ready-to-use :class:`crewai.Agent`.
        """
        tools = tools or []
        logger.info("QualityCheckerAgent.create — model=%s tools=%d", model, len(tools))
        return Agent(
            role="Chief Quality Assurance Editor",
            goal=(
                "Rigorously evaluate research reports for factual accuracy, "
                "citation correctness, completeness of coverage, and logical flow. "
                "Score each dimension 0–25 (max total 100). "
                "Provide specific, actionable improvement suggestions. "
                "Return all findings in the required structured JSON format."
            ),
            backstory=(
                "You are a meticulous fact-checker and senior editor with 20+ years "
                "of experience in investigative journalism and academic peer review. "
                "You have a zero-tolerance policy for unsupported claims and broken "
                "citations, but you give fair, constructive feedback that writers can "
                "act on immediately. You are the last line of defence before a report "
                "is published."
            ),
            tools=tools,
            llm=model,
            verbose=verbose,
            max_iter=max_iter,
            allow_delegation=False,
            memory=True,
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class QualityCheckerRunner:
    """End-to-end quality-check orchestrator.

    Retry strategy:
    - Up to ``max_retries`` attempts if the crew raises an exception.
    - Parse failure falls back to a synthetic low-scoring report (no retry).

    Revision loop:
    - ``run_with_revision_loop`` accepts any ``revise_fn(report, notes) → str``
      callback and iterates until ``overall_score >= quality_threshold`` or
      ``max_rounds`` is exhausted.

    Example::

        checker = QualityCheckerRunner(quality_threshold=70)
        report = checker.check(
            report_content=my_report,
            research_output=research,
            topic="Quantum computing",
        )
        print(report.to_markdown())
    """

    def __init__(
        self,
        *,
        model: str = "gpt-4o",
        max_retries: int = 3,
        quality_threshold: int = _DEFAULT_THRESHOLD,
        verbose: bool = False,
    ) -> None:
        """Initialise the runner.

        Args:
            model:             OpenAI model used by the agent.
            max_retries:       Crew-level retry limit on exceptions.
            quality_threshold: Score (0–100) below which revision is required.
            verbose:           Pass to CrewAI for chain-of-thought logging.
        """
        if not 0 <= quality_threshold <= 100:
            raise ValueError(f"quality_threshold must be 0–100, got {quality_threshold}")
        self._model = model
        self._max_retries = max_retries
        self._threshold = quality_threshold
        self._verbose = verbose
        logger.info(
            "QualityCheckerRunner init — model=%s threshold=%d max_retries=%d",
            model, quality_threshold, max_retries,
        )

    # ------------------------------------------------------------------
    # Public API — single check
    # ------------------------------------------------------------------

    def check(
        self,
        report_content: str,
        research_output: ResearchOutput | None = None,
        topic: str = "",
    ) -> QualityReport:
        """Quality-check *report_content* and return a :class:`QualityReport`.

        Args:
            report_content:  The Markdown report text to evaluate.
            research_output: Optional original research data for cross-checking.
            topic:           Research topic string (for context and logging).

        Returns:
            A :class:`QualityReport` (``success=False`` on unrecoverable error).
        """
        logger.info("QualityCheckerRunner.check — topic=%r", topic or "(no topic)")
        start = time.perf_counter()

        last_error = ""
        for attempt in range(1, self._max_retries + 1):
            logger.info("Check attempt %d/%d", attempt, self._max_retries)
            try:
                raw = self._execute_crew(report_content, research_output, topic)
                report = self._parse_output(
                    raw=raw,
                    topic=topic,
                    duration=time.perf_counter() - start,
                    retry_count=attempt - 1,
                )
                report.success = True
                logger.info(
                    "Check complete — score=%d verdict=%s attempt=%d",
                    report.overall_score, report.verdict, attempt,
                )
                return report
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                logger.error("Check attempt %d failed: %s", attempt, exc, exc_info=True)

        # All retries exhausted
        return QualityReport(
            topic=topic,
            overall_score=0,
            dimensions=self._default_dimensions(),
            issues=[
                QualityIssue(
                    dimension="system",
                    severity=IssueSeverity.CRITICAL,
                    description=f"Quality check failed after {self._max_retries} attempts.",
                    suggestion="Retry the quality check.",
                )
            ],
            top_suggestions=["Retry the quality check — all attempts failed."],
            verdict="REVISION_REQUIRED",
            revision_required=True,
            revision_prompt=f"Quality check failed: {last_error}",
            success=False,
            error=last_error,
            duration_secs=time.perf_counter() - start,
            retry_count=self._max_retries,
        )

    # ------------------------------------------------------------------
    # Public API — revision loop
    # ------------------------------------------------------------------

    def run_with_revision_loop(
        self,
        report_content: str,
        revise_fn: Callable[[str, str], str],
        research_output: ResearchOutput | None = None,
        topic: str = "",
        max_rounds: int = 3,
    ) -> tuple[str, QualityReport]:
        """Check quality and iteratively revise until approved or max rounds reached.

        Each round:
        1. Run ``check()`` on the current report.
        2. If approved (``score >= threshold``), stop immediately.
        3. Otherwise call ``revise_fn(current_report, revision_notes)`` to get a
           revised report, then loop.

        Args:
            report_content:  Initial Markdown report.
            revise_fn:       Callable ``(report: str, notes: str) -> str`` that
                             returns a revised version of the report.
            research_output: Original research data for context.
            topic:           Research topic string.
            max_rounds:      Maximum total check-and-revise cycles.

        Returns:
            ``(final_report_content, final_quality_report)`` after the loop ends.
        """
        if max_rounds < 1:
            raise ValueError(f"max_rounds must be ≥ 1, got {max_rounds}")

        current_report = report_content
        final_quality: QualityReport | None = None

        for round_num in range(1, max_rounds + 1):
            logger.info(
                "Revision loop — round %d/%d for topic=%r", round_num, max_rounds, topic
            )

            quality = self.check(
                report_content=current_report,
                research_output=research_output,
                topic=topic,
            )
            final_quality = quality

            if not quality.revision_required:
                logger.info(
                    "Report approved at round %d — score=%d",
                    round_num, quality.overall_score,
                )
                break

            if round_num >= max_rounds:
                logger.warning(
                    "Max rounds (%d) reached — final score=%d (threshold=%d)",
                    max_rounds, quality.overall_score, self._threshold,
                )
                break

            revision_notes = quality.revision_prompt or "\n".join(
                f"- {s}" for s in quality.top_suggestions[:5]
            )
            logger.info(
                "Score %d below threshold %d — requesting revision (round %d)",
                quality.overall_score, self._threshold, round_num,
            )
            try:
                current_report = revise_fn(current_report, revision_notes)
                logger.info("Revision produced %d chars", len(current_report))
            except Exception as exc:  # noqa: BLE001
                logger.error("revise_fn failed at round %d: %s", round_num, exc)
                break

        # final_quality is always set because max_rounds >= 1
        assert final_quality is not None
        return current_report, final_quality

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _execute_crew(
        self,
        report_content: str,
        research_output: ResearchOutput | None,
        topic: str,
    ) -> str:
        """Build the CrewAI agent + task, kick off, return raw output text."""
        task_description = self._build_task_description(
            report_content, research_output, topic
        )
        agent = QualityCheckerAgent.create(
            tools=[],
            model=self._model,
            verbose=self._verbose,
        )
        task = Task(
            description=task_description,
            expected_output=(
                f"A {_QUALITY_JSON_START} / {_QUALITY_JSON_END} JSON block "
                "containing dimension scores, issues, top suggestions, "
                "verdict, and revision_prompt."
            ),
            agent=agent,
        )
        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=self._verbose,
        )
        logger.debug("Kicking off quality check crew — topic=%r", topic)
        crew_output = crew.kickoff()
        return crew_output.raw if hasattr(crew_output, "raw") else str(crew_output)

    def _build_task_description(
        self,
        report_content: str,
        research_output: ResearchOutput | None,
        topic: str,
    ) -> str:
        """Compose the LLM task prompt for quality evaluation."""
        # Truncate long inputs to stay within context limits
        report_snippet = (
            report_content[:4000] + "\n\n[... report truncated ...]"
            if len(report_content) > 4000
            else report_content
        )

        # Build research context block
        context_lines: list[str] = []
        if research_output:
            if research_output.key_findings:
                context_lines.append("**Key Findings from Research:**")
                for i, finding in enumerate(research_output.key_findings[:8], 1):
                    context_lines.append(f"  {i}. {finding}")
                context_lines.append("")

            if research_output.sources:
                context_lines.append("**Sources Retrieved:**")
                for src in research_output.top_sources[:8]:
                    context_lines.append(
                        f"  - [{src.title or src.url}]({src.url})"
                        f" (relevance: {src.relevance_score:.2f})"
                    )
        else:
            context_lines.append(
                "*No original research data provided — evaluate the report on its own merits.*"
            )

        context_block = "\n".join(context_lines)

        return f"""You are a senior quality assurance editor. Evaluate the research report below.

## Topic
{topic or "(not specified)"}

## Report Under Review
{report_snippet}

## Original Research Context
{context_block}

---

## Evaluation Criteria
Score each dimension **0–25**. The four dimension scores sum to the overall score (max 100).

### 1. Factual Consistency (0–25)
- Every significant claim is supported by the research evidence above.
- No hallucinated statistics, dates, or attributed quotes.
- Contrasting viewpoints are acknowledged where they exist.
- Numbers match the original source data.

### 2. Citation Accuracy (0–25)
- All significant claims include a source reference or URL.
- URLs are plausible and match the claim context.
- No circular, broken, or obviously fabricated citations.
- Attribution is correct (right claim linked to right source).

### 3. Completeness of Coverage (0–25)
- All key findings from the research are addressed in the report.
- The executive summary captures the most important points.
- No major topic areas from the research are omitted.
- Recommendations are grounded in the findings.

### 4. Logical Flow (0–25)
- Clear and logical narrative from introduction to conclusion.
- Smooth transitions between sections.
- Conclusions follow naturally from the evidence presented.
- Appropriate depth and balance across sections.

---

## Issue Severity Guide
- **CRITICAL** — factual error, fabricated citation, or major logical contradiction (must fix before publication).
- **MAJOR**    — significant coverage gap or structural flaw that weakens the report.
- **MINOR**    — style, phrasing, or small omission that can be improved.

---

## Required Output Format

You MUST include the evaluation inside the delimiters shown below.
Do NOT omit or rename the delimiters.

{_QUALITY_JSON_START}
{{
  "dimensions": {{
    "factual_consistency": {{
      "score": <int 0-25>,
      "issues": ["<specific issue description>"],
      "suggestions": ["<actionable fix>"]
    }},
    "citation_accuracy": {{
      "score": <int 0-25>,
      "issues": [],
      "suggestions": []
    }},
    "completeness": {{
      "score": <int 0-25>,
      "issues": [],
      "suggestions": []
    }},
    "logical_flow": {{
      "score": <int 0-25>,
      "issues": [],
      "suggestions": []
    }}
  }},
  "top_suggestions": [
    "<most important fix 1>",
    "<most important fix 2>",
    "<most important fix 3>"
  ],
  "verdict": "<APPROVED or REVISION_REQUIRED>",
  "revision_prompt": "<targeted instructions for the report writer, or null if approved>"
}}
{_QUALITY_JSON_END}

You may include additional commentary before or after the JSON block.
"""

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------

    def _parse_output(
        self,
        raw: str,
        topic: str,
        duration: float,
        retry_count: int,
    ) -> QualityReport:
        """Parse the agent's raw text into a :class:`QualityReport`.

        Strategy:
        1. Extract JSON between ``QUALITY_JSON_START`` … ``QUALITY_JSON_END``.
        2. Fallback: largest ``{…}`` blob in the text.
        3. Last resort: return a synthetic low-quality report with the raw text.
        """
        logger.debug("Parsing quality check output (%d chars)", len(raw))
        data = self._extract_json(raw)

        if data is None:
            logger.warning("JSON parse failed — returning fallback quality report")
            return self._fallback_report(topic, raw, duration, retry_count)

        dimensions = self._parse_dimensions(data.get("dimensions", {}))
        overall_score = sum(d.score for d in dimensions)
        issues = self._collect_issues(dimensions)

        top_suggestions: list[str] = data.get("top_suggestions", [])
        if not top_suggestions:
            # Auto-generate from dimension suggestions if agent omitted them
            top_suggestions = [
                s
                for d in dimensions
                for s in d.suggestions
            ][:5]

        revision_required = overall_score < self._threshold
        verdict: str = data.get("verdict", "")
        if verdict not in ("APPROVED", "REVISION_REQUIRED"):
            verdict = "REVISION_REQUIRED" if revision_required else "APPROVED"

        # Build revision prompt
        revision_prompt: str | None = data.get("revision_prompt")
        if revision_required and not revision_prompt:
            revision_prompt = self._generate_revision_prompt(
                overall_score, dimensions, top_suggestions
            )
        if not revision_required:
            revision_prompt = None

        logger.info(
            "Parsed quality report — score=%d verdict=%s issues=%d",
            overall_score, verdict, len(issues),
        )
        return QualityReport(
            topic=topic,
            overall_score=overall_score,
            dimensions=dimensions,
            issues=issues,
            top_suggestions=top_suggestions[:5],
            verdict=verdict,  # type: ignore[arg-type]
            revision_required=revision_required,
            revision_prompt=revision_prompt,
            duration_secs=duration,
            retry_count=retry_count,
        )

    def _parse_dimensions(self, raw_dims: dict[str, Any]) -> list[QualityDimension]:
        """Convert the agent's dimension dict into ``QualityDimension`` objects."""
        result: list[QualityDimension] = []
        for meta in _DIMENSION_META:
            dim_name = meta["name"].value  # enum → string
            raw = raw_dims.get(dim_name, {}) if isinstance(raw_dims, dict) else {}
            result.append(
                QualityDimension(
                    name=dim_name,
                    display_name=meta["display"],
                    score=raw.get("score", 0) if isinstance(raw, dict) else 0,
                    issues=raw.get("issues", []) if isinstance(raw, dict) else [],
                    suggestions=raw.get("suggestions", []) if isinstance(raw, dict) else [],
                )
            )
        return result

    @staticmethod
    def _collect_issues(dimensions: list[QualityDimension]) -> list[QualityIssue]:
        """Build a flat ``QualityIssue`` list from per-dimension issue strings."""
        issues: list[QualityIssue] = []
        for dim in dimensions:
            severity = (
                IssueSeverity.CRITICAL
                if dim.percentage < 40
                else IssueSeverity.MAJOR
                if dim.percentage < 70
                else IssueSeverity.MINOR
            )
            for desc in dim.issues:
                suggestion = dim.suggestions[len(issues) % max(len(dim.suggestions), 1)] \
                    if dim.suggestions else ""
                issues.append(
                    QualityIssue(
                        dimension=dim.name,
                        severity=severity,
                        description=desc,
                        suggestion=suggestion,
                    )
                )
        return issues

    def _generate_revision_prompt(
        self,
        score: int,
        dimensions: list[QualityDimension],
        suggestions: list[str],
    ) -> str:
        """Build a targeted revision prompt when the agent doesn't provide one."""
        lines = [
            f"The report scored {score}/100 (threshold: {self._threshold}).",
            "Please address the following issues before resubmission:",
            "",
        ]
        for dim in dimensions:
            if not dim.passed:
                lines.append(
                    f"**{dim.display_name}** ({dim.score}/{dim.max_score}):"
                )
                for issue in dim.issues[:3]:
                    lines.append(f"  - {issue}")
                for sug in dim.suggestions[:2]:
                    lines.append(f"  → Fix: {sug}")
                lines.append("")

        if suggestions:
            lines.append("**Priority actions:**")
            for i, sug in enumerate(suggestions[:5], 1):
                lines.append(f"  {i}. {sug}")

        return "\n".join(lines)

    @staticmethod
    def _extract_json(text: str) -> dict | None:
        """Try to extract a JSON dict from *text* via two strategies."""
        # Strategy 1: delimited fence
        pattern = (
            rf"{re.escape(_QUALITY_JSON_START)}\s*({{.*?}})\s*{re.escape(_QUALITY_JSON_END)}"
        )
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError as exc:
                logger.debug("Fenced JSON parse failed: %s", exc)

        # Strategy 2: largest {...} blob
        candidates = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
        for candidate in sorted(candidates, key=len, reverse=True):
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue

        return None

    def _fallback_report(
        self,
        topic: str,
        raw: str,
        duration: float,
        retry_count: int,
    ) -> QualityReport:
        """Return a synthetic report when JSON parsing fails entirely."""
        dims = self._default_dimensions()
        return QualityReport(
            topic=topic,
            overall_score=0,
            dimensions=dims,
            issues=[
                QualityIssue(
                    dimension="system",
                    severity=IssueSeverity.CRITICAL,
                    description="Quality check output could not be parsed.",
                    suggestion="Retry the quality check.",
                )
            ],
            top_suggestions=[
                "Quality check output was unparseable — retry the check.",
                raw[:200] if raw else "",
            ],
            verdict="REVISION_REQUIRED",
            revision_required=True,
            revision_prompt="Quality check output could not be parsed. Please resubmit.",
            success=False,
            error="JSON parse failure — no structured output from agent.",
            duration_secs=duration,
            retry_count=retry_count,
        )

    @staticmethod
    def _default_dimensions() -> list[QualityDimension]:
        """Return four zero-scored dimensions for error / fallback cases."""
        return [
            QualityDimension(name=meta["name"].value, display_name=meta["display"])
            for meta in _DIMENSION_META
        ]
