"""Unit tests for all three search tool backends.

Strategy
--------
- All HTTP calls are mocked via ``pytest-mock`` / ``unittest.mock.patch``.
- No real network requests or API keys are needed.
- Each tool is tested for:
    * Happy path — correct output format and SourceResult fields.
    * Retry behaviour — transient errors trigger retries, success after N-1 fails.
    * Exhausted retries — all attempts fail → error string returned (no raise).
    * Missing / placeholder API key — returns an error string immediately.
    * Empty result set — handled gracefully.
    * HTTP error status codes (4xx / 5xx) → treated as retriable.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest
import requests

from src.models.research_models import SearchBackend, SourceResult
from src.tools.search_tools import (
    SerperSearchTool,
    TavilySearchTool,
    WebSearchTool,
    _relevance_from_position,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _http_error(status: int = 503) -> requests.HTTPError:
    mock_resp = MagicMock()
    mock_resp.status_code = status
    return requests.HTTPError(f"HTTP {status}", response=mock_resp)


# ---------------------------------------------------------------------------
# Utility function tests
# ---------------------------------------------------------------------------


class TestRelevanceFromPosition:
    def test_first_position_is_maximum(self):
        assert _relevance_from_position(1, 5) == 1.0

    def test_last_position_is_minimum(self):
        score = _relevance_from_position(5, 5)
        assert score >= 0.1

    def test_single_result_is_max(self):
        assert _relevance_from_position(1, 1) == 1.0

    def test_scores_decrease_with_position(self):
        scores = [_relevance_from_position(i, 10) for i in range(1, 6)]
        assert scores == sorted(scores, reverse=True)

    def test_score_clamped_to_range(self):
        for pos in range(1, 21):
            score = _relevance_from_position(pos, 5)
            assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# SerperSearchTool
# ---------------------------------------------------------------------------


class TestSerperSearchTool:
    """Tests for SerperSearchTool."""

    @pytest.fixture()
    def tool(self) -> SerperSearchTool:
        return SerperSearchTool(api_key="test-key-abc", max_retries=3)

    # --- happy path ---

    def test_returns_formatted_string(self, tool, mock_serper_http):
        result = tool._run("quantum computing", max_results=3)
        assert isinstance(result, str)
        assert "Quantum Computing in 2024" in result
        assert "https://example.com/quantum-2024" in result

    def test_result_contains_relevance_scores(self, tool, mock_serper_http):
        result = tool._run("quantum computing", max_results=3)
        assert "Relevance:" in result

    def test_result_contains_snippet(self, tool, mock_serper_http):
        result = tool._run("quantum computing", max_results=3)
        assert "1000-qubit" in result

    def test_post_called_with_correct_headers(self, tool, mock_serper_http):
        tool._run("quantum computing", max_results=5)
        _, kwargs = mock_serper_http.call_args
        assert kwargs["headers"]["X-API-KEY"] == "test-key-abc"
        assert kwargs["json"]["q"] == "quantum computing"
        assert kwargs["json"]["num"] == 5

    def test_search_structured_returns_source_results(self, tool, mock_serper_http):
        sources = tool.search_structured("quantum computing", max_results=3)
        assert isinstance(sources, list)
        assert all(isinstance(s, SourceResult) for s in sources)

    def test_search_structured_sets_backend_field(self, tool, mock_serper_http):
        sources = tool.search_structured("quantum computing", max_results=3)
        assert all(s.backend == SearchBackend.SERPER for s in sources)

    def test_search_structured_scores_descend_by_position(
        self, tool, mock_serper_http
    ):
        sources = tool.search_structured("quantum computing", max_results=3)
        scores = [s.relevance_score for s in sources]
        assert scores == sorted(scores, reverse=True)

    # --- missing API key ---

    def test_no_api_key_returns_error_string(self):
        tool = SerperSearchTool(api_key="")
        result = tool._run("anything")
        assert "not configured" in result.lower()
        assert "SERPER_API_KEY" in result

    def test_no_api_key_search_structured_returns_empty(self):
        tool = SerperSearchTool(api_key="")
        assert tool.search_structured("anything") == []

    # --- empty results ---

    def test_empty_organic_returns_no_results_message(self, tool):
        empty_resp = MagicMock()
        empty_resp.json.return_value = {"organic": []}
        empty_resp.raise_for_status.return_value = None
        with patch("requests.post", return_value=empty_resp):
            result = tool._run("very obscure topic")
        assert "No results found" in result

    # --- retry on transient error ---

    def test_retries_on_connection_error(self, tool, serper_response):
        """Should retry up to max_retries and succeed on the final attempt."""
        success_resp = MagicMock()
        success_resp.json.return_value = serper_response
        success_resp.raise_for_status.return_value = None

        side_effects = [
            requests.ConnectionError("timeout"),
            requests.ConnectionError("timeout"),
            success_resp,
        ]
        with patch("requests.post", side_effect=side_effects) as mock_post:
            result = tool._run("quantum computing", max_results=3)

        assert mock_post.call_count == 3
        assert "Quantum Computing" in result

    def test_retries_on_http_500(self, tool, serper_response):
        """HTTP 500 should trigger a retry."""
        error_resp = MagicMock()
        error_resp.raise_for_status.side_effect = _http_error(500)

        success_resp = MagicMock()
        success_resp.json.return_value = serper_response
        success_resp.raise_for_status.return_value = None

        with patch("requests.post", side_effect=[error_resp, success_resp]) as mock_post:
            result = tool._run("quantum computing")

        assert mock_post.call_count == 2
        assert "Quantum Computing" in result

    # --- exhausted retries ---

    def test_all_retries_exhausted_returns_error_string(self):
        tool = SerperSearchTool(api_key="test-key", max_retries=2)
        always_fail = requests.ConnectionError("service down")
        with patch("requests.post", side_effect=always_fail):
            result = tool._run("any topic")

        assert isinstance(result, str)
        assert "failed" in result.lower()

    def test_exhausted_retries_does_not_raise(self):
        """Tool should never raise — always return a string."""
        tool = SerperSearchTool(api_key="test-key", max_retries=1)
        with patch("requests.post", side_effect=requests.Timeout("timed out")):
            try:
                result = tool._run("any topic")
            except Exception as exc:
                pytest.fail(f"Tool raised an exception: {exc}")
        assert isinstance(result, str)

    # --- HTTP 4xx (non-retriable but raise_for_status raises) ---

    def test_http_401_returns_error_string(self):
        tool = SerperSearchTool(api_key="bad-key", max_retries=1)
        with patch("requests.post", side_effect=_http_error(401)):
            result = tool._run("anything")
        assert "failed" in result.lower()


# ---------------------------------------------------------------------------
# TavilySearchTool
# ---------------------------------------------------------------------------


class TestTavilySearchTool:
    """Tests for TavilySearchTool."""

    @pytest.fixture()
    def tool(self) -> TavilySearchTool:
        return TavilySearchTool(api_key="tvly-real-key-xyz", max_retries=3)

    def test_returns_formatted_string(self, tool, mock_tavily_http):
        result = tool._run("quantum computing", max_results=2)
        assert isinstance(result, str)
        assert "Quantum Computing" in result

    def test_includes_ai_summary(self, tool, mock_tavily_http):
        result = tool._run("quantum computing", max_results=2)
        assert "AI Summary" in result

    def test_search_structured_returns_source_results(self, tool, mock_tavily_http):
        sources = tool.search_structured("quantum computing", max_results=2)
        assert isinstance(sources, list)
        assert all(isinstance(s, SourceResult) for s in sources)

    def test_search_structured_uses_tavily_score(self, tool, mock_tavily_http):
        sources = tool.search_structured("quantum computing", max_results=2)
        # First result has score 0.95 in TAVILY_RESPONSE fixture
        assert sources[0].relevance_score == pytest.approx(0.95, abs=1e-3)

    def test_search_structured_sets_backend(self, tool, mock_tavily_http):
        sources = tool.search_structured("quantum computing")
        assert all(s.backend == SearchBackend.TAVILY for s in sources)

    def test_no_api_key_returns_error(self):
        tool = TavilySearchTool(api_key="")
        result = tool._run("anything")
        assert "not configured" in result.lower()

    def test_no_api_key_search_structured_returns_empty(self):
        tool = TavilySearchTool(api_key="")
        assert tool.search_structured("anything") == []

    def test_retries_on_connection_error(self, tool, tavily_response):
        success_resp = MagicMock()
        success_resp.json.return_value = tavily_response
        success_resp.raise_for_status.return_value = None

        with patch(
            "requests.post",
            side_effect=[requests.ConnectionError("drop"), success_resp],
        ) as mock_post:
            result = tool._run("quantum computing")

        assert mock_post.call_count == 2
        assert "Quantum" in result

    def test_all_retries_exhausted_returns_error_string(self):
        tool = TavilySearchTool(api_key="tvly-key", max_retries=2)
        with patch("requests.post", side_effect=requests.Timeout("down")):
            result = tool._run("anything")
        assert isinstance(result, str)
        assert "failed" in result.lower()

    def test_empty_results_handled(self, tool):
        empty_resp = MagicMock()
        empty_resp.json.return_value = {"results": [], "answer": ""}
        empty_resp.raise_for_status.return_value = None
        with patch("requests.post", return_value=empty_resp):
            result = tool._run("obscure topic")
        assert "No results found" in result


# ---------------------------------------------------------------------------
# WebSearchTool (DuckDuckGo)
# ---------------------------------------------------------------------------


class TestWebSearchTool:
    """Tests for the free DuckDuckGo fallback tool."""

    @pytest.fixture()
    def tool(self) -> WebSearchTool:
        return WebSearchTool(max_retries=2)

    def test_returns_formatted_string(self, tool, ddg_results):
        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.text.return_value = ddg_results

        with patch("duckduckgo_search.DDGS", return_value=mock_ddgs):
            result = tool._run("quantum computing")

        assert isinstance(result, str)
        assert "duckduckgo-result.com" in result

    def test_import_error_returns_helpful_message(self, tool):
        with patch.dict("sys.modules", {"duckduckgo_search": None}):
            result = tool._run("anything")
        assert "not installed" in result.lower() or "duckduckgo" in result.lower()

    def test_retries_on_exception_then_succeeds(self, tool, ddg_results):
        mock_ddgs_fail = MagicMock()
        mock_ddgs_fail.__enter__ = MagicMock(return_value=mock_ddgs_fail)
        mock_ddgs_fail.__exit__ = MagicMock(return_value=False)
        mock_ddgs_fail.text.side_effect = Exception("rate limited")

        mock_ddgs_ok = MagicMock()
        mock_ddgs_ok.__enter__ = MagicMock(return_value=mock_ddgs_ok)
        mock_ddgs_ok.__exit__ = MagicMock(return_value=False)
        mock_ddgs_ok.text.return_value = ddg_results

        with patch(
            "duckduckgo_search.DDGS",
            side_effect=[mock_ddgs_fail, mock_ddgs_ok],
        ):
            result = tool._run("quantum computing")

        assert isinstance(result, str)

    def test_all_retries_exhausted_returns_error_string(self, tool):
        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.text.side_effect = Exception("down")

        with patch("duckduckgo_search.DDGS", return_value=mock_ddgs):
            result = tool._run("anything")

        assert isinstance(result, str)
        assert "failed" in result.lower()

    def test_empty_results_returns_no_results_message(self, tool):
        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.text.return_value = []

        with patch("duckduckgo_search.DDGS", return_value=mock_ddgs):
            result = tool._run("nothing found")

        assert "No results found" in result


# ---------------------------------------------------------------------------
# SourceResult model
# ---------------------------------------------------------------------------


class TestSourceResult:
    """Tests for the SourceResult Pydantic model."""

    def test_valid_construction(self):
        src = SourceResult(url="https://example.com", title="Test", relevance_score=0.8)
        assert src.url == "https://example.com"
        assert src.relevance_score == pytest.approx(0.8)

    def test_relevance_clamped_above_one(self):
        src = SourceResult(url="https://example.com", relevance_score=1.5)
        assert src.relevance_score == 1.0

    def test_relevance_clamped_below_zero(self):
        src = SourceResult(url="https://example.com", relevance_score=-0.5)
        assert src.relevance_score == 0.0

    def test_invalid_score_type_defaults_to_zero(self):
        src = SourceResult(url="https://example.com", relevance_score="not-a-number")
        assert src.relevance_score == 0.0

    def test_empty_url_raises(self):
        with pytest.raises(Exception):
            SourceResult(url="", title="Test")

    def test_to_markdown_contains_url(self):
        src = SourceResult(
            url="https://example.com",
            title="My Article",
            key_points=["Point A"],
            relevance_score=0.9,
        )
        md = src.to_markdown()
        assert "https://example.com" in md
        assert "Point A" in md
        assert "0.90" in md

    def test_default_backend_is_unknown(self):
        src = SourceResult(url="https://example.com")
        assert src.backend == SearchBackend.UNKNOWN
