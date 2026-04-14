"""Unit tests for QualityCheckerAgent (factory) and QualityCheckerRunner (orchestrator).

All CrewAI / OpenAI calls are mocked — no real LLM API calls are made.

Key mocking rules:
- ``src.agents.quality_checker_agent.Agent`` — the local name after
  ``from crewai import Agent``; patching ``crewai.Agent`` would not work.
- Same applies to ``Task`` and ``Crew``.
- Factory tests assert on the kwargs passed to the Agent *constructor* rather
  than inspecting the live crewai.Agent object (which would init the LLM).

Test groups
-----------
TestQualityCheckerAgentFactory   — agent constructor called with correct kwargs.
TestQualityDimension             — Pydantic model properties (percentage, passed).
TestQualityReport                — QualityReport properties and to_markdown().
TestQualityCheckerRunnerInit     — threshold validation and attribute defaults.
TestQualityCheckerRunnerParsing  — JSON extraction + output parsing in isolation.
TestQualityCheckerRunnerFullCheck — end-to-end check() with mocked Crew.kickoff().
TestRevisionLoop                 — run_with_revision_loop() branching logic.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from unittest.mock import MagicMock, call, patch

import pytest

from src.agents.quality_checker_agent import (
    QualityCheckerAgent,
    QualityCheckerRunner,
    _QUALITY_JSON_END,
    _QUALITY_JSON_START,
    _DEFAULT_THRESHOLD,
)
from src.models.quality_models import (
    DimensionName,
    IssueSeverity,
    QualityDimension,
    QualityIssue,
    QualityReport,
)

# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------

TOPIC = "AI in healthcare"

GOOD_QC_PAYLOAD: dict = {
    "dimensions": {
        "factual_consistency": {
            "score": 22,
            "issues": ["One minor unsupported claim"],
            "suggestions": ["Add citation for the 2023 study"],
        },
        "citation_accuracy": {
            "score": 20,
            "issues": [],
            "suggestions": [],
        },
        "completeness": {
            "score": 18,
            "issues": ["Drug discovery section is thin"],
            "suggestions": ["Expand on AI drug discovery examples"],
        },
        "logical_flow": {
            "score": 21,
            "issues": [],
            "suggestions": [],
        },
    },
    "top_suggestions": [
        "Add citation for the 2023 study",
        "Expand on AI drug discovery examples",
    ],
    "verdict": "APPROVED",
    "revision_prompt": None,
}

# Overall score = 22 + 20 + 18 + 21 = 81 → APPROVED

LOW_SCORE_PAYLOAD: dict = {
    "dimensions": {
        "factual_consistency": {
            "score": 8,
            "issues": ["Hallucinated statistic", "Wrong date"],
            "suggestions": ["Remove unsupported claim", "Fix date to 2022"],
        },
        "citation_accuracy": {
            "score": 6,
            "issues": ["Broken URL in section 2"],
            "suggestions": ["Replace with working URL"],
        },
        "completeness": {
            "score": 5,
            "issues": ["Missing ethics discussion"],
            "suggestions": ["Add ethics section"],
        },
        "logical_flow": {
            "score": 10,
            "issues": ["Abrupt conclusion"],
            "suggestions": ["Extend the conclusion"],
        },
    },
    "top_suggestions": [
        "Remove unsupported claims",
        "Fix broken citations",
        "Add ethics section",
    ],
    "verdict": "REVISION_REQUIRED",
    "revision_prompt": "Please fix the factual errors and missing citations.",
}

# Overall score = 8 + 6 + 5 + 10 = 29 → REVISION_REQUIRED


def _make_crew_output(raw: str) -> MagicMock:
    mock = MagicMock()
    mock.raw = raw
    return mock


def _fenced_json(payload: dict) -> str:
    return (
        "Quality analysis:\n\n"
        f"{_QUALITY_JSON_START}\n"
        f"{json.dumps(payload, indent=2)}\n"
        f"{_QUALITY_JSON_END}\n"
        "End of review."
    )


GOOD_RAW_OUTPUT = _fenced_json(GOOD_QC_PAYLOAD)
LOW_SCORE_RAW_OUTPUT = _fenced_json(LOW_SCORE_PAYLOAD)


@contextmanager
def _mock_qc_crew(raw_output: str = GOOD_RAW_OUTPUT):
    """Patch Agent, Task, Crew inside quality_checker_agent so no LLM calls happen."""
    crew_output = _make_crew_output(raw_output)
    with (
        patch("src.agents.quality_checker_agent.Agent") as MockAgent,
        patch("src.agents.quality_checker_agent.Task") as MockTask,
        patch("src.agents.quality_checker_agent.Crew") as MockCrew,
    ):
        mock_crew_instance = MockCrew.return_value
        mock_crew_instance.kickoff.return_value = crew_output
        yield MockAgent, MockTask, MockCrew


# ---------------------------------------------------------------------------
# TestQualityCheckerAgentFactory
# ---------------------------------------------------------------------------


class TestQualityCheckerAgentFactory:
    def test_create_passes_role(self):
        with patch("src.agents.quality_checker_agent.Agent") as MockAgent:
            MockAgent.return_value = MagicMock()
            QualityCheckerAgent.create(model="gpt-4o")
            kwargs = MockAgent.call_args.kwargs
            assert kwargs["role"] == "Chief Quality Assurance Editor"

    def test_create_passes_model(self):
        with patch("src.agents.quality_checker_agent.Agent") as MockAgent:
            MockAgent.return_value = MagicMock()
            QualityCheckerAgent.create(model="gpt-4o-mini")
            kwargs = MockAgent.call_args.kwargs
            assert kwargs["llm"] == "gpt-4o-mini"

    def test_create_no_delegation(self):
        with patch("src.agents.quality_checker_agent.Agent") as MockAgent:
            MockAgent.return_value = MagicMock()
            QualityCheckerAgent.create()
            kwargs = MockAgent.call_args.kwargs
            assert kwargs["allow_delegation"] is False

    def test_create_empty_tools_by_default(self):
        with patch("src.agents.quality_checker_agent.Agent") as MockAgent:
            MockAgent.return_value = MagicMock()
            QualityCheckerAgent.create()
            kwargs = MockAgent.call_args.kwargs
            assert kwargs["tools"] == []

    def test_create_accepts_tools(self):
        with patch("src.agents.quality_checker_agent.Agent") as MockAgent:
            MockAgent.return_value = MagicMock()
            fake_tool = MagicMock()
            QualityCheckerAgent.create(tools=[fake_tool])
            kwargs = MockAgent.call_args.kwargs
            assert kwargs["tools"] == [fake_tool]

    def test_create_verbose_passed_through(self):
        with patch("src.agents.quality_checker_agent.Agent") as MockAgent:
            MockAgent.return_value = MagicMock()
            QualityCheckerAgent.create(verbose=False)
            kwargs = MockAgent.call_args.kwargs
            assert kwargs["verbose"] is False

    def test_create_max_iter_passed_through(self):
        with patch("src.agents.quality_checker_agent.Agent") as MockAgent:
            MockAgent.return_value = MagicMock()
            QualityCheckerAgent.create(max_iter=5)
            kwargs = MockAgent.call_args.kwargs
            assert kwargs["max_iter"] == 5


# ---------------------------------------------------------------------------
# TestQualityDimension
# ---------------------------------------------------------------------------


class TestQualityDimension:
    def test_percentage_full_score(self):
        dim = QualityDimension(name="factual_consistency", display_name="FC", score=25)
        assert dim.percentage == 100.0

    def test_percentage_zero(self):
        dim = QualityDimension(name="factual_consistency", display_name="FC", score=0)
        assert dim.percentage == 0.0

    def test_percentage_partial(self):
        dim = QualityDimension(name="factual_consistency", display_name="FC", score=20)
        assert dim.percentage == 80.0

    def test_passed_at_threshold(self):
        # 18/25 = 72% → passed
        dim = QualityDimension(name="x", display_name="X", score=18)
        assert dim.passed is True

    def test_failed_below_threshold(self):
        # 17/25 = 68% → not passed
        dim = QualityDimension(name="x", display_name="X", score=17)
        assert dim.passed is False

    def test_score_clamped_above_max(self):
        dim = QualityDimension(name="x", display_name="X", score=999)
        assert dim.score == 25

    def test_score_clamped_below_zero(self):
        dim = QualityDimension(name="x", display_name="X", score=-5)
        assert dim.score == 0

    def test_non_numeric_score_defaults_to_zero(self):
        dim = QualityDimension(name="x", display_name="X", score="bad")  # type: ignore[arg-type]
        assert dim.score == 0

    def test_issues_default_empty(self):
        dim = QualityDimension(name="x", display_name="X")
        assert dim.issues == []

    def test_suggestions_default_empty(self):
        dim = QualityDimension(name="x", display_name="X")
        assert dim.suggestions == []


# ---------------------------------------------------------------------------
# TestQualityReport
# ---------------------------------------------------------------------------


class TestQualityReport:
    def _approved_report(self) -> QualityReport:
        dims = [
            QualityDimension(name="factual_consistency", display_name="FC", score=22),
            QualityDimension(name="citation_accuracy", display_name="CA", score=20),
            QualityDimension(name="completeness", display_name="CO", score=18),
            QualityDimension(name="logical_flow", display_name="LF", score=21),
        ]
        return QualityReport(
            topic="AI in healthcare",
            overall_score=81,
            dimensions=dims,
            issues=[],
            top_suggestions=["Add citations"],
            verdict="APPROVED",
            revision_required=False,
            success=True,
        )

    def _low_score_report(self) -> QualityReport:
        dims = [
            QualityDimension(
                name="factual_consistency",
                display_name="FC",
                score=8,
                issues=["Hallucinated stat"],
            ),
            QualityDimension(name="citation_accuracy", display_name="CA", score=6),
            QualityDimension(name="completeness", display_name="CO", score=5),
            QualityDimension(name="logical_flow", display_name="LF", score=10),
        ]
        issues = [
            QualityIssue(
                dimension="factual_consistency",
                severity=IssueSeverity.CRITICAL,
                description="Hallucinated stat",
                suggestion="Remove it",
            )
        ]
        return QualityReport(
            topic="AI in healthcare",
            overall_score=29,
            dimensions=dims,
            issues=issues,
            verdict="REVISION_REQUIRED",
            revision_required=True,
            revision_prompt="Fix the errors.",
            success=True,
        )

    def test_has_critical_issues_true(self):
        report = self._low_score_report()
        assert report.has_critical_issues is True

    def test_has_critical_issues_false(self):
        report = self._approved_report()
        assert report.has_critical_issues is False

    def test_dimension_map_keys(self):
        report = self._approved_report()
        assert set(report.dimension_map.keys()) == {
            "factual_consistency",
            "citation_accuracy",
            "completeness",
            "logical_flow",
        }

    def test_get_dimension_returns_correct(self):
        report = self._approved_report()
        dim = report.get_dimension("factual_consistency")
        assert dim is not None
        assert dim.score == 22

    def test_get_dimension_missing_returns_none(self):
        report = self._approved_report()
        assert report.get_dimension("nonexistent") is None

    def test_failed_dimensions_all_pass(self):
        report = self._approved_report()
        assert report.failed_dimensions == []

    def test_failed_dimensions_some_fail(self):
        report = self._low_score_report()
        # All four dimensions score < 70%
        assert len(report.failed_dimensions) == 4

    def test_overall_score_clamped(self):
        report = QualityReport(overall_score=999)
        assert report.overall_score == 100

    def test_overall_score_negative_clamped(self):
        report = QualityReport(overall_score=-1)
        assert report.overall_score == 0

    def test_to_markdown_contains_topic(self):
        report = self._approved_report()
        md = report.to_markdown()
        assert "AI in healthcare" in md

    def test_to_markdown_contains_score(self):
        report = self._approved_report()
        md = report.to_markdown()
        assert "81" in md

    def test_to_markdown_approved_verdict(self):
        report = self._approved_report()
        md = report.to_markdown()
        assert "APPROVED" in md

    def test_to_markdown_revision_required_verdict(self):
        report = self._low_score_report()
        md = report.to_markdown()
        assert "REVISION_REQUIRED" in md

    def test_to_markdown_includes_revision_prompt(self):
        report = self._low_score_report()
        md = report.to_markdown()
        assert "Fix the errors" in md

    def test_to_markdown_no_revision_prompt_when_approved(self):
        report = self._approved_report()
        md = report.to_markdown()
        assert "Revision Instructions" not in md

    def test_to_markdown_contains_dimension_table(self):
        report = self._approved_report()
        md = report.to_markdown()
        assert "Dimension Scores" in md
        assert "FC" in md or "Factual" in md

    def test_to_markdown_shows_issues(self):
        report = self._low_score_report()
        md = report.to_markdown()
        assert "Hallucinated stat" in md


# ---------------------------------------------------------------------------
# TestQualityCheckerRunnerInit
# ---------------------------------------------------------------------------


class TestQualityCheckerRunnerInit:
    def test_default_threshold(self):
        runner = QualityCheckerRunner()
        assert runner._threshold == _DEFAULT_THRESHOLD

    def test_custom_threshold(self):
        runner = QualityCheckerRunner(quality_threshold=80)
        assert runner._threshold == 80

    def test_threshold_zero_allowed(self):
        runner = QualityCheckerRunner(quality_threshold=0)
        assert runner._threshold == 0

    def test_threshold_100_allowed(self):
        runner = QualityCheckerRunner(quality_threshold=100)
        assert runner._threshold == 100

    def test_threshold_below_zero_raises(self):
        with pytest.raises(ValueError, match="quality_threshold"):
            QualityCheckerRunner(quality_threshold=-1)

    def test_threshold_above_100_raises(self):
        with pytest.raises(ValueError, match="quality_threshold"):
            QualityCheckerRunner(quality_threshold=101)

    def test_default_model(self):
        runner = QualityCheckerRunner()
        assert runner._model == "gpt-4o"

    def test_custom_model(self):
        runner = QualityCheckerRunner(model="gpt-4o-mini")
        assert runner._model == "gpt-4o-mini"

    def test_default_max_retries(self):
        runner = QualityCheckerRunner()
        assert runner._max_retries == 3


# ---------------------------------------------------------------------------
# TestQualityCheckerRunnerParsing
# ---------------------------------------------------------------------------


class TestQualityCheckerRunnerParsing:
    def setup_method(self):
        self.runner = QualityCheckerRunner()

    # --- _extract_json ---

    def test_extract_json_from_fence(self):
        payload = {"dimensions": {}, "verdict": "APPROVED"}
        text = _fenced_json(payload)
        result = QualityCheckerRunner._extract_json(text)
        assert result == payload

    def test_extract_json_fallback_to_blob(self):
        payload = {"verdict": "APPROVED", "dimensions": {}}
        text = f"Some text. {json.dumps(payload)} more text."
        result = QualityCheckerRunner._extract_json(text)
        assert result is not None
        assert result["verdict"] == "APPROVED"

    def test_extract_json_returns_none_for_no_json(self):
        assert QualityCheckerRunner._extract_json("No JSON here at all.") is None

    def test_extract_json_prefers_fenced_over_blob(self):
        fenced_payload = {"source": "fence", "dimensions": {}}
        blob_payload = {"source": "blob"}
        text = (
            f"blob: {json.dumps(blob_payload)} "
            f"{_QUALITY_JSON_START}\n{json.dumps(fenced_payload)}\n{_QUALITY_JSON_END}"
        )
        result = QualityCheckerRunner._extract_json(text)
        assert result["source"] == "fence"

    # --- _parse_dimensions ---

    def test_parse_dimensions_all_four_present(self):
        dims = self.runner._parse_dimensions(GOOD_QC_PAYLOAD["dimensions"])
        assert len(dims) == 4

    def test_parse_dimensions_names_ordered(self):
        dims = self.runner._parse_dimensions(GOOD_QC_PAYLOAD["dimensions"])
        assert dims[0].name == DimensionName.FACTUAL_CONSISTENCY.value
        assert dims[1].name == DimensionName.CITATION_ACCURACY.value
        assert dims[2].name == DimensionName.COMPLETENESS.value
        assert dims[3].name == DimensionName.LOGICAL_FLOW.value

    def test_parse_dimensions_scores_correct(self):
        dims = self.runner._parse_dimensions(GOOD_QC_PAYLOAD["dimensions"])
        assert dims[0].score == 22
        assert dims[1].score == 20
        assert dims[2].score == 18
        assert dims[3].score == 21

    def test_parse_dimensions_missing_key_defaults_to_zero(self):
        dims = self.runner._parse_dimensions({})
        assert all(d.score == 0 for d in dims)
        assert len(dims) == 4

    def test_parse_dimensions_issues_preserved(self):
        dims = self.runner._parse_dimensions(GOOD_QC_PAYLOAD["dimensions"])
        fc = next(d for d in dims if d.name == "factual_consistency")
        assert "One minor unsupported claim" in fc.issues

    def test_parse_dimensions_suggestions_preserved(self):
        dims = self.runner._parse_dimensions(GOOD_QC_PAYLOAD["dimensions"])
        fc = next(d for d in dims if d.name == "factual_consistency")
        assert "Add citation for the 2023 study" in fc.suggestions

    # --- _collect_issues ---

    def test_collect_issues_empty_when_no_issues(self):
        dims = [
            QualityDimension(name="factual_consistency", display_name="FC", score=22),
            QualityDimension(name="citation_accuracy", display_name="CA", score=20),
            QualityDimension(name="completeness", display_name="CO", score=18),
            QualityDimension(name="logical_flow", display_name="LF", score=21),
        ]
        issues = QualityCheckerRunner._collect_issues(dims)
        assert issues == []

    def test_collect_issues_critical_severity_below_40pct(self):
        # score=8 → 32% → CRITICAL
        dims = [QualityDimension(name="factual_consistency", display_name="FC", score=8,
                                 issues=["Bad claim"])]
        issues = QualityCheckerRunner._collect_issues(dims)
        assert issues[0].severity == IssueSeverity.CRITICAL

    def test_collect_issues_major_severity_below_70pct(self):
        # score=15 → 60% → MAJOR
        dims = [QualityDimension(name="citation_accuracy", display_name="CA", score=15,
                                 issues=["Missing citation"])]
        issues = QualityCheckerRunner._collect_issues(dims)
        assert issues[0].severity == IssueSeverity.MAJOR

    def test_collect_issues_minor_severity_at_or_above_70pct(self):
        # score=20 → 80% → MINOR
        dims = [QualityDimension(name="completeness", display_name="CO", score=20,
                                 issues=["Small omission"])]
        issues = QualityCheckerRunner._collect_issues(dims)
        assert issues[0].severity == IssueSeverity.MINOR

    def test_collect_issues_dimension_name_set(self):
        dims = [QualityDimension(name="logical_flow", display_name="LF", score=5,
                                 issues=["Bad flow"])]
        issues = QualityCheckerRunner._collect_issues(dims)
        assert issues[0].dimension == "logical_flow"

    def test_collect_issues_multiple_issues_per_dimension(self):
        dims = [QualityDimension(name="completeness", display_name="CO", score=5,
                                 issues=["Issue A", "Issue B", "Issue C"])]
        issues = QualityCheckerRunner._collect_issues(dims)
        assert len(issues) == 3

    # --- _parse_output (integration of all sub-parsers) ---

    def test_parse_output_approved_score(self):
        report = self.runner._parse_output(
            raw=GOOD_RAW_OUTPUT,
            topic=TOPIC,
            duration=1.0,
            retry_count=0,
        )
        assert report.overall_score == 81  # 22+20+18+21

    def test_parse_output_verdict_approved(self):
        report = self.runner._parse_output(
            raw=GOOD_RAW_OUTPUT,
            topic=TOPIC,
            duration=1.0,
            retry_count=0,
        )
        assert report.verdict == "APPROVED"
        assert report.revision_required is False

    def test_parse_output_revision_required_low_score(self):
        runner = QualityCheckerRunner(quality_threshold=70)
        report = runner._parse_output(
            raw=LOW_SCORE_RAW_OUTPUT,
            topic=TOPIC,
            duration=0.5,
            retry_count=0,
        )
        assert report.overall_score == 29  # 8+6+5+10
        assert report.revision_required is True
        assert report.verdict == "REVISION_REQUIRED"

    def test_parse_output_revision_prompt_preserved_from_agent(self):
        runner = QualityCheckerRunner(quality_threshold=70)
        report = runner._parse_output(
            raw=LOW_SCORE_RAW_OUTPUT,
            topic=TOPIC,
            duration=0.5,
            retry_count=0,
        )
        assert "factual errors" in (report.revision_prompt or "").lower() or \
               report.revision_prompt is not None

    def test_parse_output_no_revision_prompt_when_approved(self):
        report = self.runner._parse_output(
            raw=GOOD_RAW_OUTPUT,
            topic=TOPIC,
            duration=1.0,
            retry_count=0,
        )
        assert report.revision_prompt is None

    def test_parse_output_top_suggestions_preserved(self):
        report = self.runner._parse_output(
            raw=GOOD_RAW_OUTPUT,
            topic=TOPIC,
            duration=1.0,
            retry_count=0,
        )
        assert len(report.top_suggestions) >= 1
        assert any("citation" in s.lower() for s in report.top_suggestions)

    def test_parse_output_topic_set(self):
        report = self.runner._parse_output(
            raw=GOOD_RAW_OUTPUT,
            topic=TOPIC,
            duration=1.0,
            retry_count=0,
        )
        assert report.topic == TOPIC

    def test_parse_output_duration_set(self):
        report = self.runner._parse_output(
            raw=GOOD_RAW_OUTPUT,
            topic=TOPIC,
            duration=2.5,
            retry_count=0,
        )
        assert report.duration_secs >= 0

    def test_parse_output_fallback_on_no_json(self):
        report = self.runner._parse_output(
            raw="No valid JSON here at all.",
            topic=TOPIC,
            duration=0.1,
            retry_count=0,
        )
        # Fallback sets success=False and score=0
        assert report.success is False
        assert report.overall_score == 0

    def test_parse_output_fallback_report_has_system_issue(self):
        report = self.runner._parse_output(
            raw="Garbage output",
            topic=TOPIC,
            duration=0.1,
            retry_count=0,
        )
        assert any(i.dimension == "system" for i in report.issues)

    def test_parse_output_auto_generates_verdict_when_missing(self):
        """If agent emits a garbage verdict, runner picks one from score."""
        payload = {**GOOD_QC_PAYLOAD, "verdict": "MAYBE"}
        report = self.runner._parse_output(
            raw=_fenced_json(payload),
            topic=TOPIC,
            duration=1.0,
            retry_count=0,
        )
        # Score 81 >= 70 threshold → APPROVED
        assert report.verdict == "APPROVED"

    def test_parse_output_top_suggestions_capped_at_five(self):
        payload = {**GOOD_QC_PAYLOAD, "top_suggestions": [f"Suggestion {i}" for i in range(10)]}
        report = self.runner._parse_output(
            raw=_fenced_json(payload),
            topic=TOPIC,
            duration=1.0,
            retry_count=0,
        )
        assert len(report.top_suggestions) <= 5


# ---------------------------------------------------------------------------
# TestQualityCheckerRunnerFullCheck
# ---------------------------------------------------------------------------


class TestQualityCheckerRunnerFullCheck:
    def test_check_returns_quality_report(self):
        runner = QualityCheckerRunner()
        with _mock_qc_crew(GOOD_RAW_OUTPUT):
            result = runner.check(report_content="## Report\n\nContent.", topic=TOPIC)
        assert isinstance(result, QualityReport)

    def test_check_success_true_on_good_output(self):
        runner = QualityCheckerRunner()
        with _mock_qc_crew(GOOD_RAW_OUTPUT):
            result = runner.check(report_content="## Report", topic=TOPIC)
        assert result.success is True

    def test_check_approved_when_high_score(self):
        runner = QualityCheckerRunner(quality_threshold=70)
        with _mock_qc_crew(GOOD_RAW_OUTPUT):
            result = runner.check(report_content="## Report", topic=TOPIC)
        assert result.overall_score == 81
        assert result.revision_required is False

    def test_check_revision_required_when_low_score(self):
        runner = QualityCheckerRunner(quality_threshold=70)
        with _mock_qc_crew(LOW_SCORE_RAW_OUTPUT):
            result = runner.check(report_content="## Report", topic=TOPIC)
        assert result.revision_required is True

    def test_check_uses_topic(self):
        runner = QualityCheckerRunner()
        with _mock_qc_crew(GOOD_RAW_OUTPUT):
            result = runner.check(report_content="Report", topic="Custom Topic")
        assert result.topic == "Custom Topic"

    def test_check_crew_kicked_off_once(self):
        runner = QualityCheckerRunner()
        with _mock_qc_crew(GOOD_RAW_OUTPUT) as (_, __, MockCrew):
            runner.check(report_content="Report", topic=TOPIC)
        MockCrew.return_value.kickoff.assert_called_once()

    def test_check_retries_on_exception(self):
        runner = QualityCheckerRunner(max_retries=3)
        with (
            patch("src.agents.quality_checker_agent.Agent"),
            patch("src.agents.quality_checker_agent.Task"),
            patch("src.agents.quality_checker_agent.Crew") as MockCrew,
        ):
            # Fail first time, succeed second time
            good_output = _make_crew_output(GOOD_RAW_OUTPUT)
            MockCrew.return_value.kickoff.side_effect = [
                RuntimeError("API error"),
                good_output,
            ]
            result = runner.check(report_content="Report", topic=TOPIC)
        assert result.success is True
        assert MockCrew.return_value.kickoff.call_count == 2

    def test_check_returns_failure_after_all_retries_exhausted(self):
        runner = QualityCheckerRunner(max_retries=2)
        with (
            patch("src.agents.quality_checker_agent.Agent"),
            patch("src.agents.quality_checker_agent.Task"),
            patch("src.agents.quality_checker_agent.Crew") as MockCrew,
        ):
            MockCrew.return_value.kickoff.side_effect = RuntimeError("always fails")
            result = runner.check(report_content="Report", topic=TOPIC)
        assert result.success is False
        assert result.error is not None
        assert result.overall_score == 0

    def test_check_without_research_output(self):
        runner = QualityCheckerRunner()
        with _mock_qc_crew(GOOD_RAW_OUTPUT):
            result = runner.check(report_content="Report", topic=TOPIC, research_output=None)
        assert isinstance(result, QualityReport)

    def test_check_with_research_output(self):
        from src.models.research_models import ResearchOutput, SearchBackend

        runner = QualityCheckerRunner()
        research = ResearchOutput(
            topic=TOPIC,
            key_findings=["Finding 1", "Finding 2"],
            sources=[],
            success=True,
            backend=SearchBackend.SERPER,
        )
        with _mock_qc_crew(GOOD_RAW_OUTPUT):
            result = runner.check(
                report_content="Report",
                research_output=research,
                topic=TOPIC,
            )
        assert isinstance(result, QualityReport)

    def test_check_task_description_includes_topic(self):
        runner = QualityCheckerRunner()
        with _mock_qc_crew(GOOD_RAW_OUTPUT) as (_, MockTask, __):
            runner.check(report_content="Report body", topic="Space exploration")
        task_kwargs = MockTask.call_args.kwargs
        assert "Space exploration" in task_kwargs["description"]

    def test_check_task_description_includes_report_content(self):
        runner = QualityCheckerRunner()
        content = "## Executive Summary\n\nContent goes here."
        with _mock_qc_crew(GOOD_RAW_OUTPUT) as (_, MockTask, __):
            runner.check(report_content=content, topic=TOPIC)
        task_kwargs = MockTask.call_args.kwargs
        assert "Executive Summary" in task_kwargs["description"]

    def test_check_long_report_truncated_in_task(self):
        runner = QualityCheckerRunner()
        long_report = "x" * 6000
        with _mock_qc_crew(GOOD_RAW_OUTPUT) as (_, MockTask, __):
            runner.check(report_content=long_report, topic=TOPIC)
        task_kwargs = MockTask.call_args.kwargs
        # The truncation marker should appear
        assert "truncated" in task_kwargs["description"]

    def test_check_short_report_not_truncated(self):
        runner = QualityCheckerRunner()
        short_report = "Short content"
        with _mock_qc_crew(GOOD_RAW_OUTPUT) as (_, MockTask, __):
            runner.check(report_content=short_report, topic=TOPIC)
        task_kwargs = MockTask.call_args.kwargs
        assert "truncated" not in task_kwargs["description"]


# ---------------------------------------------------------------------------
# TestRevisionLoop
# ---------------------------------------------------------------------------


class TestRevisionLoop:
    def test_revision_loop_stops_when_approved(self):
        runner = QualityCheckerRunner(quality_threshold=70)
        revise_fn = MagicMock(return_value="Revised report")
        with _mock_qc_crew(GOOD_RAW_OUTPUT):  # score=81 → APPROVED immediately
            final_report, quality = runner.run_with_revision_loop(
                report_content="Initial report",
                revise_fn=revise_fn,
                topic=TOPIC,
                max_rounds=3,
            )
        # revise_fn should NOT be called because report is approved on round 1
        revise_fn.assert_not_called()
        assert quality.revision_required is False

    def test_revision_loop_calls_revise_fn_when_score_low(self):
        runner = QualityCheckerRunner(quality_threshold=70)
        revise_fn = MagicMock(return_value="Revised report")

        # Always return low score (never approved within max_rounds)
        with _mock_qc_crew(LOW_SCORE_RAW_OUTPUT):
            final_report, quality = runner.run_with_revision_loop(
                report_content="Bad report",
                revise_fn=revise_fn,
                topic=TOPIC,
                max_rounds=2,
            )
        # revise_fn called once (after round 1; round 2 exits because max reached)
        revise_fn.assert_called_once()

    def test_revision_loop_returns_final_report_string(self):
        runner = QualityCheckerRunner(quality_threshold=70)
        revise_fn = MagicMock(return_value="Improved report")

        with _mock_qc_crew(LOW_SCORE_RAW_OUTPUT):
            final_report, _ = runner.run_with_revision_loop(
                report_content="Bad report",
                revise_fn=revise_fn,
                topic=TOPIC,
                max_rounds=2,
            )
        assert isinstance(final_report, str)

    def test_revision_loop_returns_quality_report(self):
        runner = QualityCheckerRunner(quality_threshold=70)
        revise_fn = MagicMock(return_value="Revised")
        with _mock_qc_crew(GOOD_RAW_OUTPUT):
            _, quality = runner.run_with_revision_loop(
                report_content="Report",
                revise_fn=revise_fn,
                topic=TOPIC,
            )
        assert isinstance(quality, QualityReport)

    def test_revision_loop_max_rounds_zero_raises(self):
        runner = QualityCheckerRunner()
        with pytest.raises(ValueError, match="max_rounds"):
            runner.run_with_revision_loop(
                report_content="Report",
                revise_fn=lambda r, n: r,
                max_rounds=0,
            )

    def test_revision_loop_passes_revision_notes_to_revise_fn(self):
        runner = QualityCheckerRunner(quality_threshold=70)
        captured_notes: list[str] = []

        def revise_fn(report: str, notes: str) -> str:
            captured_notes.append(notes)
            return "Improved"

        with _mock_qc_crew(LOW_SCORE_RAW_OUTPUT):
            runner.run_with_revision_loop(
                report_content="Bad report",
                revise_fn=revise_fn,
                topic=TOPIC,
                max_rounds=2,
            )

        assert len(captured_notes) == 1
        # Notes should not be empty
        assert captured_notes[0].strip() != ""

    def test_revision_loop_revise_fn_exception_stops_loop(self):
        runner = QualityCheckerRunner(quality_threshold=70)

        def failing_revise(report: str, notes: str) -> str:
            raise RuntimeError("Writer crashed")

        with _mock_qc_crew(LOW_SCORE_RAW_OUTPUT):
            # Should not propagate the exception — loop exits gracefully
            final_report, quality = runner.run_with_revision_loop(
                report_content="Bad report",
                revise_fn=failing_revise,
                topic=TOPIC,
                max_rounds=3,
            )
        assert isinstance(final_report, str)
        assert isinstance(quality, QualityReport)

    def test_revision_loop_approved_on_second_round(self):
        runner = QualityCheckerRunner(quality_threshold=70)
        revise_fn = MagicMock(return_value="Better report")

        # Simulate: round 1 → low score, round 2 → high score
        call_count = 0

        def _check_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return QualityReport(
                    topic=TOPIC,
                    overall_score=40,
                    verdict="REVISION_REQUIRED",
                    revision_required=True,
                    revision_prompt="Fix please",
                    success=True,
                )
            return QualityReport(
                topic=TOPIC,
                overall_score=82,
                verdict="APPROVED",
                revision_required=False,
                success=True,
            )

        with patch.object(runner, "check", side_effect=_check_side_effect):
            _, quality = runner.run_with_revision_loop(
                report_content="Initial",
                revise_fn=revise_fn,
                topic=TOPIC,
                max_rounds=3,
            )

        revise_fn.assert_called_once()  # Only revised once
        assert quality.revision_required is False
        assert quality.overall_score == 82


# ---------------------------------------------------------------------------
# TestGenerateRevisionPrompt
# ---------------------------------------------------------------------------


class TestGenerateRevisionPrompt:
    def test_revision_prompt_includes_score(self):
        runner = QualityCheckerRunner(quality_threshold=70)
        dims = [
            QualityDimension(name="factual_consistency", display_name="FC", score=8,
                             issues=["Bad claim"], suggestions=["Fix it"]),
        ]
        prompt = runner._generate_revision_prompt(score=30, dimensions=dims, suggestions=[])
        assert "30" in prompt
        assert "70" in prompt

    def test_revision_prompt_includes_failed_dimension(self):
        runner = QualityCheckerRunner(quality_threshold=70)
        dims = [
            QualityDimension(name="completeness", display_name="Completeness of Coverage",
                             score=10, issues=["Missing section"]),
        ]
        prompt = runner._generate_revision_prompt(score=40, dimensions=dims, suggestions=[])
        assert "Completeness" in prompt
        assert "Missing section" in prompt

    def test_revision_prompt_excludes_passing_dimensions(self):
        runner = QualityCheckerRunner(quality_threshold=70)
        dims = [
            QualityDimension(name="factual_consistency", display_name="FC", score=8,
                             issues=["Bad claim"]),
            QualityDimension(name="logical_flow", display_name="LF", score=24,
                             issues=["Minor nit"]),
        ]
        prompt = runner._generate_revision_prompt(score=32, dimensions=dims, suggestions=[])
        # LF passes (96%) — its issues should not appear
        assert "LF" not in prompt

    def test_revision_prompt_includes_priority_suggestions(self):
        runner = QualityCheckerRunner()
        dims = [
            QualityDimension(name="citation_accuracy", display_name="CA", score=5,
                             issues=["Broken URL"])
        ]
        suggestions = ["Add citations", "Fix broken links"]
        prompt = runner._generate_revision_prompt(score=20, dimensions=dims,
                                                   suggestions=suggestions)
        assert "Add citations" in prompt


# ---------------------------------------------------------------------------
# TestDefaultDimensions
# ---------------------------------------------------------------------------


class TestDefaultDimensions:
    def test_default_dimensions_returns_four(self):
        dims = QualityCheckerRunner._default_dimensions()
        assert len(dims) == 4

    def test_default_dimensions_all_zero(self):
        dims = QualityCheckerRunner._default_dimensions()
        assert all(d.score == 0 for d in dims)

    def test_default_dimensions_names_correct(self):
        dims = QualityCheckerRunner._default_dimensions()
        names = [d.name for d in dims]
        assert "factual_consistency" in names
        assert "citation_accuracy" in names
        assert "completeness" in names
        assert "logical_flow" in names
