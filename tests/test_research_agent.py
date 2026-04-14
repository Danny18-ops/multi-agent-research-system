"""Unit tests for ResearchAgent (factory) and ResearchRunner (orchestrator).

All CrewAI / OpenAI calls are mocked — no real LLM API calls are made.

Key mocking rules applied here:
- `src.agents.research_agent.Agent` — the name *in the module under test*,
  not `crewai.Agent`, because the code does ``from crewai import Agent``.
- `src.agents.research_agent.Crew`  — same reason.
- Factory tests verify the kwargs passed to the Agent constructor rather
  than inspecting the live crewai.Agent object (which would init the LLM).

Test groups
-----------
TestResearchAgentFactory     — agent constructor is called with correct kwargs.
TestResearchRunnerBackendSelection — Serper > Tavily > DDG priority.
TestResearchRunnerOutputParsing    — JSON extraction + output parsing in isolation.
TestResearchRunnerFullRun          — end-to-end run with mocked Crew.kickoff.
TestResearchOutput                 — Pydantic model properties and serialisation.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from src.agents.research_agent import (
    ResearchAgent,
    ResearchRunner,
    _OUTPUT_FENCE_END,
    _OUTPUT_FENCE_START,
)
from src.models.research_models import ResearchOutput, SearchBackend, SourceResult

# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------

TOPIC = "Quantum computing breakthroughs 2024"


def _make_crew_output(raw: str) -> MagicMock:
    mock = MagicMock()
    mock.raw = raw
    return mock


def _fenced_json(payload: dict) -> str:
    return (
        "Some analysis text.\n\n"
        f"{_OUTPUT_FENCE_START}\n"
        f"{json.dumps(payload, indent=2)}\n"
        f"{_OUTPUT_FENCE_END}\n"
        "Concluding remarks."
    )


GOOD_PAYLOAD = {
    "search_queries": ["quantum computing 2024", "quantum milestones"],
    "key_findings": ["Finding A", "Finding B", "Finding C"],
    "sources": [
        {
            "url": "https://example.com/a",
            "title": "Article A",
            "snippet": "Snippet A",
            "key_points": ["Point 1", "Point 2"],
            "relevance_score": 0.92,
            "position": 1,
        },
        {
            "url": "https://example.com/b",
            "title": "Article B",
            "snippet": "Snippet B",
            "key_points": ["Point 3"],
            "relevance_score": 0.75,
            "position": 2,
        },
    ],
}

EMPTY_PAYLOAD = {**GOOD_PAYLOAD, "sources": [], "key_findings": []}


@contextmanager
def _mock_crew(kickoff_side_effect=None, kickoff_return=None):
    """Patch Agent, Task, and Crew in the module under test.

    - Agent  → MagicMock (avoids LLM initialisation).
    - Task   → MagicMock (avoids Pydantic validation that rejects a MagicMock agent).
    - Crew   → MagicMock with kickoff() returning the supplied canned output.

    Yields ``(MockCrew, crew_instance)`` so callers can assert call counts.
    """
    with (
        patch("src.agents.research_agent.Agent"),
        patch("src.agents.research_agent.Task"),
        patch("src.agents.research_agent.Crew") as MockCrew,
    ):
        instance = MockCrew.return_value
        if kickoff_side_effect is not None:
            instance.kickoff.side_effect = kickoff_side_effect
        elif kickoff_return is not None:
            instance.kickoff.return_value = kickoff_return
        yield MockCrew, instance


# ---------------------------------------------------------------------------
# ResearchAgent factory tests
# ---------------------------------------------------------------------------


class TestResearchAgentFactory:
    """Verify that create() calls crewai.Agent with the expected keyword args."""

    @pytest.fixture()
    def mock_tool(self) -> MagicMock:
        tool = MagicMock()
        tool.name = "mock_search"
        return tool

    def test_agent_has_correct_role(self, mock_tool):
        with patch("src.agents.research_agent.Agent") as MockAgent:
            ResearchAgent.create(tools=[mock_tool])
        kwargs = MockAgent.call_args.kwargs
        assert "Research" in kwargs["role"]

    def test_agent_has_correct_goal_mentions_research(self, mock_tool):
        with patch("src.agents.research_agent.Agent") as MockAgent:
            ResearchAgent.create(tools=[mock_tool])
        kwargs = MockAgent.call_args.kwargs
        assert "research" in kwargs["goal"].lower()

    def test_agent_receives_tools(self, mock_tool):
        with patch("src.agents.research_agent.Agent") as MockAgent:
            ResearchAgent.create(tools=[mock_tool])
        kwargs = MockAgent.call_args.kwargs
        assert mock_tool in kwargs["tools"]

    def test_agent_respects_max_iter(self, mock_tool):
        with patch("src.agents.research_agent.Agent") as MockAgent:
            ResearchAgent.create(tools=[mock_tool], max_iter=7)
        kwargs = MockAgent.call_args.kwargs
        assert kwargs["max_iter"] == 7

    def test_agent_delegation_disabled(self, mock_tool):
        with patch("src.agents.research_agent.Agent") as MockAgent:
            ResearchAgent.create(tools=[mock_tool])
        kwargs = MockAgent.call_args.kwargs
        assert kwargs["allow_delegation"] is False

    def test_agent_memory_enabled(self, mock_tool):
        with patch("src.agents.research_agent.Agent") as MockAgent:
            ResearchAgent.create(tools=[mock_tool])
        kwargs = MockAgent.call_args.kwargs
        assert kwargs["memory"] is True

    def test_agent_llm_uses_supplied_model(self, mock_tool):
        with patch("src.agents.research_agent.Agent") as MockAgent:
            ResearchAgent.create(tools=[mock_tool], model="gpt-4o-mini")
        kwargs = MockAgent.call_args.kwargs
        assert kwargs["llm"] == "gpt-4o-mini"

    def test_create_returns_agent_instance(self, mock_tool):
        with patch("src.agents.research_agent.Agent") as MockAgent:
            result = ResearchAgent.create(tools=[mock_tool])
        assert result is MockAgent.return_value


# ---------------------------------------------------------------------------
# ResearchRunner — backend selection
# ---------------------------------------------------------------------------


class TestResearchRunnerBackendSelection:
    """Verify that the correct search backend is selected from keys."""

    def _make_runner(self, serper: str = "", tavily: str = "") -> ResearchRunner:
        # Pass explicit strings (not None) so the constructor skips config loading
        return ResearchRunner(serper_api_key=serper, tavily_api_key=tavily, max_retries=1)

    def test_prefers_serper_when_both_keys_present(self):
        runner = self._make_runner(serper="real-serper-key", tavily="real-tavily-key")
        assert runner._backend == SearchBackend.SERPER

    def test_uses_tavily_when_no_serper(self):
        runner = self._make_runner(serper="", tavily="real-tavily-key")
        assert runner._backend == SearchBackend.TAVILY

    def test_falls_back_to_duckduckgo_when_no_keys(self):
        runner = self._make_runner(serper="", tavily="")
        assert runner._backend == SearchBackend.DUCKDUCKGO

    def test_placeholder_serper_key_not_used(self):
        runner = self._make_runner(serper="your-serper-api-key-here", tavily="")
        assert runner._backend == SearchBackend.DUCKDUCKGO

    def test_placeholder_tavily_key_not_used(self):
        runner = self._make_runner(serper="", tavily="your-tavily-api-key-here")
        assert runner._backend == SearchBackend.DUCKDUCKGO

    def test_serper_tool_set_correctly(self):
        from src.tools.search_tools import SerperSearchTool

        runner = self._make_runner(serper="real-key", tavily="")
        assert isinstance(runner._search_tool, SerperSearchTool)

    def test_tavily_tool_set_correctly(self):
        from src.tools.search_tools import TavilySearchTool

        # serper="" forces no-Serper path; tavily key triggers Tavily
        runner = self._make_runner(serper="", tavily="real-tavily-key")
        assert isinstance(runner._search_tool, TavilySearchTool)

    def test_ddg_tool_set_correctly(self):
        from src.tools.search_tools import WebSearchTool

        runner = self._make_runner(serper="", tavily="")
        assert isinstance(runner._search_tool, WebSearchTool)


# ---------------------------------------------------------------------------
# ResearchRunner — JSON parsing (no LLM calls needed)
# ---------------------------------------------------------------------------


class TestResearchRunnerOutputParsing:
    """Unit-test _extract_json and _parse_output in isolation."""

    @pytest.fixture()
    def runner(self) -> ResearchRunner:
        # Explicit empty strings → skip config loading → DuckDuckGo backend
        return ResearchRunner(serper_api_key="", tavily_api_key="", max_retries=1)

    # --- _extract_json ---

    def test_extracts_fenced_json(self, runner):
        text = _fenced_json(GOOD_PAYLOAD)
        result = runner._extract_json(text)
        assert result is not None
        assert result["key_findings"] == GOOD_PAYLOAD["key_findings"]

    def test_extracts_unfenced_json_blob(self, runner):
        text = "Some text. " + json.dumps(GOOD_PAYLOAD) + " More text."
        result = runner._extract_json(text)
        assert result is not None
        assert "sources" in result

    def test_returns_none_for_plain_text(self, runner):
        result = runner._extract_json("There is no JSON here, just text.")
        assert result is None

    def test_prefers_fenced_over_unfenced(self, runner):
        inner = {"search_queries": ["fence query"], "key_findings": [], "sources": []}
        # Put a decoy dict before the fenced block
        text = f'{{"decoy": 1}} ' + _fenced_json(inner)
        result = runner._extract_json(text)
        assert result["search_queries"] == ["fence query"]

    def test_handles_malformed_json_in_fence(self, runner):
        text = f"{_OUTPUT_FENCE_START}\n{{broken json\n{_OUTPUT_FENCE_END}"
        # Should fall through to strategy 2 or return None — must not raise
        result = runner._extract_json(text)
        assert result is None or isinstance(result, dict)

    # --- _parse_output ---

    def test_parse_output_populates_sources(self, runner):
        output = runner._parse_output(
            raw=_fenced_json(GOOD_PAYLOAD),
            topic=TOPIC,
            queries=[],
            duration=1.0,
            retry_count=0,
        )
        assert len(output.sources) == 2

    def test_parse_output_populates_key_findings(self, runner):
        output = runner._parse_output(
            raw=_fenced_json(GOOD_PAYLOAD),
            topic=TOPIC,
            queries=[],
            duration=1.0,
            retry_count=0,
        )
        assert output.key_findings == GOOD_PAYLOAD["key_findings"]

    def test_parse_output_deduplicates_urls(self, runner):
        dup_payload = {**GOOD_PAYLOAD, "sources": [GOOD_PAYLOAD["sources"][0]] * 2}
        output = runner._parse_output(
            raw=_fenced_json(dup_payload),
            topic=TOPIC,
            queries=[],
            duration=1.0,
            retry_count=0,
        )
        assert len(output.sources) == 1

    def test_parse_output_handles_no_json_gracefully(self, runner):
        output = runner._parse_output(
            raw="Agent could not find anything useful.",
            topic=TOPIC,
            queries=[],
            duration=0.5,
            retry_count=0,
        )
        assert isinstance(output, ResearchOutput)
        assert output.topic == TOPIC

    def test_parse_output_skips_empty_url_sources(self, runner):
        payload = {
            **GOOD_PAYLOAD,
            "sources": [
                {"url": "", "title": "No URL", "relevance_score": 0.9},
                GOOD_PAYLOAD["sources"][0],
            ],
        }
        output = runner._parse_output(
            raw=_fenced_json(payload),
            topic=TOPIC,
            queries=[],
            duration=1.0,
            retry_count=0,
        )
        assert all(src.url for src in output.sources)

    def test_parse_output_sets_backend_on_sources(self, runner):
        output = runner._parse_output(
            raw=_fenced_json(GOOD_PAYLOAD),
            topic=TOPIC,
            queries=[],
            duration=1.0,
            retry_count=0,
        )
        assert all(src.backend == runner._backend for src in output.sources)

    def test_parse_output_records_search_queries(self, runner):
        output = runner._parse_output(
            raw=_fenced_json(GOOD_PAYLOAD),
            topic=TOPIC,
            queries=[],
            duration=1.0,
            retry_count=0,
        )
        assert output.search_queries == GOOD_PAYLOAD["search_queries"]


# ---------------------------------------------------------------------------
# ResearchRunner — full run with mocked Crew
# ---------------------------------------------------------------------------


class TestResearchRunnerFullRun:
    """End-to-end run tests — Crew.kickoff returns canned output; no LLM calls."""

    @pytest.fixture()
    def runner(self) -> ResearchRunner:
        # Explicit empty strings → skip config loading → DuckDuckGo backend
        return ResearchRunner(
            serper_api_key="", tavily_api_key="", max_retries=3, verbose=False
        )

    @pytest.fixture()
    def good_output(self) -> MagicMock:
        return _make_crew_output(_fenced_json(GOOD_PAYLOAD))

    @pytest.fixture()
    def empty_output(self) -> MagicMock:
        return _make_crew_output(_fenced_json(EMPTY_PAYLOAD))

    # --- successful run ---

    def test_successful_run_returns_research_output(self, runner, good_output):
        with _mock_crew(kickoff_return=good_output):
            result = runner.run(TOPIC)
        assert isinstance(result, ResearchOutput)

    def test_successful_run_is_marked_success(self, runner, good_output):
        with _mock_crew(kickoff_return=good_output):
            result = runner.run(TOPIC)
        assert result.success is True

    def test_successful_run_has_correct_topic(self, runner, good_output):
        with _mock_crew(kickoff_return=good_output):
            result = runner.run(TOPIC)
        assert result.topic == TOPIC

    def test_successful_run_has_sources(self, runner, good_output):
        with _mock_crew(kickoff_return=good_output):
            result = runner.run(TOPIC)
        assert len(result.sources) == 2
        assert result.sources[0].url == "https://example.com/a"

    def test_successful_run_has_key_findings(self, runner, good_output):
        with _mock_crew(kickoff_return=good_output):
            result = runner.run(TOPIC)
        assert result.key_findings == GOOD_PAYLOAD["key_findings"]

    def test_successful_run_has_no_error(self, runner, good_output):
        with _mock_crew(kickoff_return=good_output):
            result = runner.run(TOPIC)
        assert result.error is None

    def test_successful_run_records_positive_duration(self, runner, good_output):
        with _mock_crew(kickoff_return=good_output):
            result = runner.run(TOPIC)
        assert result.duration_secs >= 0.0

    # --- run-level retry on exception ---

    def test_retries_on_crew_exception(self, runner, good_output):
        """First kickoff raises; second succeeds — runner must retry."""
        with _mock_crew(
            kickoff_side_effect=[RuntimeError("LLM timeout"), good_output]
        ) as (_, instance):
            result = runner.run(TOPIC)

        assert result.success is True
        assert instance.kickoff.call_count == 2

    def test_all_retries_exhausted_returns_failure(self, runner):
        with _mock_crew(kickoff_side_effect=RuntimeError("always fails")):
            result = runner.run(TOPIC)
        assert result.success is False
        assert result.error is not None

    def test_all_retries_exhausted_does_not_raise(self, runner):
        with _mock_crew(kickoff_side_effect=RuntimeError("always fails")):
            try:
                result = runner.run(TOPIC)
            except Exception as exc:
                pytest.fail(f"run() should not raise, but raised: {exc}")
        assert isinstance(result, ResearchOutput)

    def test_retry_count_incremented_on_failure(self, runner, good_output):
        with _mock_crew(
            kickoff_side_effect=[RuntimeError("err"), good_output]
        ) as (_, instance):
            result = runner.run(TOPIC)
        assert result.retry_count >= 1

    # --- quality retry (empty sources trigger another attempt) ---

    def test_quality_retry_triggered_on_empty_sources(
        self, runner, empty_output, good_output
    ):
        with _mock_crew(
            kickoff_side_effect=[empty_output, good_output]
        ) as (_, instance):
            result = runner.run(TOPIC)

        assert instance.kickoff.call_count == 2

    def test_quality_retry_second_attempt_sources_returned(
        self, runner, empty_output, good_output
    ):
        with _mock_crew(kickoff_side_effect=[empty_output, good_output]):
            result = runner.run(TOPIC)
        assert result.sources

    def test_all_empty_runs_marked_not_success(self, runner, empty_output):
        with _mock_crew(kickoff_return=empty_output):
            result = runner.run(TOPIC)
        assert result.success is False

    # --- backend field propagated to output ---

    def test_backend_field_set_on_output(self, runner, good_output):
        with _mock_crew(kickoff_return=good_output):
            result = runner.run(TOPIC)
        assert result.backend == SearchBackend.DUCKDUCKGO  # no keys → DDG

    def test_serper_backend_propagated(self, good_output):
        runner = ResearchRunner(
            serper_api_key="real-serper-key", tavily_api_key="", max_retries=1
        )
        with _mock_crew(kickoff_return=good_output):
            result = runner.run(TOPIC)
        assert result.backend == SearchBackend.SERPER

    def test_tavily_backend_propagated(self, good_output):
        runner = ResearchRunner(
            serper_api_key="", tavily_api_key="real-tavily-key", max_retries=1
        )
        with _mock_crew(kickoff_return=good_output):
            result = runner.run(TOPIC)
        assert result.backend == SearchBackend.TAVILY


# ---------------------------------------------------------------------------
# ResearchOutput model tests
# ---------------------------------------------------------------------------


class TestResearchOutput:
    """Tests for ResearchOutput properties, computed fields, and serialisation."""

    @pytest.fixture()
    def output_with_sources(self) -> ResearchOutput:
        return ResearchOutput(
            topic="Test topic",
            sources=[
                SourceResult(url="https://a.com", relevance_score=0.9),
                SourceResult(url="https://b.com", relevance_score=0.5),
                SourceResult(url="https://c.com", relevance_score=0.7),
            ],
            key_findings=["Finding 1", "Finding 2"],
            success=True,
            backend=SearchBackend.SERPER,
        )

    def test_top_sources_sorted_by_relevance_descending(self, output_with_sources):
        scores = [s.relevance_score for s in output_with_sources.top_sources]
        assert scores == sorted(scores, reverse=True)

    def test_avg_relevance_computed_correctly(self, output_with_sources):
        expected = (0.9 + 0.5 + 0.7) / 3
        assert output_with_sources.avg_relevance == pytest.approx(expected, abs=1e-6)

    def test_avg_relevance_zero_when_no_sources(self):
        output = ResearchOutput(topic="Empty", success=True)
        assert output.avg_relevance == 0.0

    def test_to_markdown_contains_topic(self, output_with_sources):
        assert "Test topic" in output_with_sources.to_markdown()

    def test_to_markdown_contains_key_findings(self, output_with_sources):
        md = output_with_sources.to_markdown()
        assert "Finding 1" in md
        assert "Finding 2" in md

    def test_to_markdown_contains_source_urls(self, output_with_sources):
        md = output_with_sources.to_markdown()
        assert "https://a.com" in md
        assert "https://b.com" in md

    def test_to_markdown_shows_error_when_failed(self):
        output = ResearchOutput(
            topic="Failed", success=False, error="Rate limit exceeded"
        )
        assert "Rate limit exceeded" in output.to_markdown()

    def test_serialise_to_dict(self, output_with_sources):
        data = output_with_sources.model_dump()
        assert data["topic"] == "Test topic"
        assert len(data["sources"]) == 3

    def test_roundtrip_json(self, output_with_sources):
        json_str = output_with_sources.model_dump_json()
        restored = ResearchOutput.model_validate_json(json_str)
        assert restored.topic == output_with_sources.topic
        assert len(restored.sources) == len(output_with_sources.sources)
        assert restored.backend == output_with_sources.backend
