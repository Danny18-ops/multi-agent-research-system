"""Shared pytest fixtures for the research system test suite.

Fixtures are organised into three groups:
- **API response mocks** — canned JSON payloads from Serper / Tavily.
- **Config fixtures**    — lightweight AppConfig with test-safe dummy keys.
- **Agent/crew mocks**   — fake CrewAI Crew output objects.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Canned API responses
# ---------------------------------------------------------------------------

SERPER_ORGANIC: list[dict[str, Any]] = [
    {
        "position": 1,
        "title": "Quantum Computing in 2024: A Breakthrough Year",
        "link": "https://example.com/quantum-2024",
        "snippet": "Researchers achieved 1000-qubit processors, marking a milestone.",
    },
    {
        "position": 2,
        "title": "IBM Unveils Next-Gen Quantum Chip",
        "link": "https://ibm.com/quantum-chip",
        "snippet": "IBM's new quantum chip reduces error rates by 50%.",
    },
    {
        "position": 3,
        "title": "Google's Quantum Supremacy Claim Revisited",
        "link": "https://nature.com/quantum-supremacy",
        "snippet": "Independent researchers verify Google's quantum supremacy experiment.",
    },
]

SERPER_RESPONSE: dict[str, Any] = {
    "searchParameters": {"q": "quantum computing breakthroughs 2024", "num": 5},
    "organic": SERPER_ORGANIC,
    "credits": 1,
}

TAVILY_RESPONSE: dict[str, Any] = {
    "query": "quantum computing breakthroughs 2024",
    "answer": "Quantum computing saw major breakthroughs in 2024 including 1000-qubit chips.",
    "results": [
        {
            "url": "https://example.com/quantum-2024",
            "title": "Quantum Computing in 2024",
            "content": "Researchers achieved 1000-qubit processors.",
            "score": 0.95,
        },
        {
            "url": "https://ibm.com/quantum-chip",
            "title": "IBM Quantum Chip",
            "content": "IBM's chip reduces error rates significantly.",
            "score": 0.87,
        },
    ],
    "response_time": 1.2,
}

DDG_RESULTS: list[dict[str, Any]] = [
    {
        "title": "Quantum Computing News",
        "href": "https://duckduckgo-result.com/article",
        "body": "Various quantum computing milestones were achieved in 2024.",
    },
]

# ---------------------------------------------------------------------------
# Structured agent JSON output (what the LLM emits)
# ---------------------------------------------------------------------------

AGENT_JSON_PAYLOAD: dict[str, Any] = {
    "search_queries": [
        "quantum computing breakthroughs 2024",
        "quantum computing milestones recent",
    ],
    "key_findings": [
        "1000-qubit processors were demonstrated for the first time.",
        "Error rates dropped by 50% with new chip architectures.",
        "Quantum supremacy claims were independently verified.",
    ],
    "sources": [
        {
            "url": "https://example.com/quantum-2024",
            "title": "Quantum Computing in 2024",
            "snippet": "Researchers achieved 1000-qubit processors.",
            "key_points": ["1000-qubit milestone", "Error rate reduction"],
            "relevance_score": 0.95,
            "position": 1,
        },
        {
            "url": "https://ibm.com/quantum-chip",
            "title": "IBM Quantum Chip",
            "snippet": "IBM's chip reduces error rates significantly.",
            "key_points": ["50% error rate reduction", "New architecture"],
            "relevance_score": 0.87,
            "position": 2,
        },
    ],
}

AGENT_RAW_OUTPUT = (
    "I researched the topic thoroughly.\n\n"
    "RESEARCH_JSON_START\n"
    + __import__("json").dumps(AGENT_JSON_PAYLOAD, indent=2)
    + "\nRESEARCH_JSON_END\n\n"
    "This concludes the research."
)

# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def serper_response() -> dict[str, Any]:
    """Canned Serper API JSON response."""
    return SERPER_RESPONSE.copy()


@pytest.fixture()
def tavily_response() -> dict[str, Any]:
    """Canned Tavily API JSON response."""
    return TAVILY_RESPONSE.copy()


@pytest.fixture()
def ddg_results() -> list[dict[str, Any]]:
    """Canned DuckDuckGo result list."""
    return DDG_RESULTS.copy()


@pytest.fixture()
def agent_raw_output() -> str:
    """Raw text output the agent would emit (includes JSON fence)."""
    return AGENT_RAW_OUTPUT


@pytest.fixture()
def agent_json_payload() -> dict[str, Any]:
    """The parsed JSON payload inside the agent's fence block."""
    return AGENT_JSON_PAYLOAD.copy()


# ---------------------------------------------------------------------------
# Mock HTTP response helper
# ---------------------------------------------------------------------------


def make_mock_response(json_data: dict, status_code: int = 200) -> MagicMock:
    """Build a :class:`unittest.mock.MagicMock` mimicking a ``requests.Response``."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = json_data
    if status_code >= 400:
        from requests import HTTPError

        mock_resp.raise_for_status.side_effect = HTTPError(
            f"HTTP {status_code}", response=mock_resp
        )
    else:
        mock_resp.raise_for_status.return_value = None
    return mock_resp


@pytest.fixture()
def mock_serper_http(serper_response):
    """Patch ``requests.post`` to return the canned Serper response."""
    with patch("requests.post", return_value=make_mock_response(serper_response)) as m:
        yield m


@pytest.fixture()
def mock_tavily_http(tavily_response):
    """Patch ``requests.post`` to return the canned Tavily response."""
    with patch("requests.post", return_value=make_mock_response(tavily_response)) as m:
        yield m


# ---------------------------------------------------------------------------
# Mock CrewAI Crew output
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_crew_output(agent_raw_output: str) -> MagicMock:
    """A mock object that behaves like a ``crewai.CrewOutput``."""
    mock = MagicMock()
    mock.raw = agent_raw_output
    return mock


@pytest.fixture()
def mock_crew(mock_crew_output: MagicMock):
    """Patch ``crewai.Crew`` so ``kickoff()`` returns canned output without LLM calls."""
    with patch("crewai.Crew") as MockCrew:
        instance = MockCrew.return_value
        instance.kickoff.return_value = mock_crew_output
        yield MockCrew


# ---------------------------------------------------------------------------
# Config fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def test_config():
    """Return a minimal AppConfig with fake-but-syntactically-valid keys."""
    from pydantic import SecretStr
    from src.config.config_loader import AppConfig

    return AppConfig(
        openai_api_key=SecretStr("sk-test-0000000000000000"),
        model_name="gpt-4o",
        serper_api_key=SecretStr("test-serper-key-abc123"),
        tavily_api_key=SecretStr(""),
        serpapi_api_key=SecretStr(""),
    )
