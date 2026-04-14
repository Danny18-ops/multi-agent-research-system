"""Research Agent and ResearchRunner.

``ResearchAgent`` is a factory for a single CrewAI ``Agent``.
``ResearchRunner`` is the high-level orchestrator that:

- Selects the right search backend from config
- Builds queries, calls the agent, and parses structured output
- Retries at three levels: HTTP (inside tools), agent-run, and result-quality
- Deduplicates sources and computes relevance scores
- Returns a fully typed :class:`~src.models.ResearchOutput`
- Logs every significant action for observability

Typical usage::

    from src.agents.research_agent import ResearchRunner

    runner = ResearchRunner()
    result = runner.run("Quantum computing breakthroughs in 2024")
    print(result.to_markdown())
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import TYPE_CHECKING

from crewai import Agent, Crew, Process, Task

from src.models.research_models import ResearchOutput, SearchBackend, SourceResult

if TYPE_CHECKING:
    from crewai.tools import BaseTool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# JSON block the agent is asked to include in its output
_OUTPUT_FENCE_START = "RESEARCH_JSON_START"
_OUTPUT_FENCE_END = "RESEARCH_JSON_END"

# Minimum sources required to consider a run successful
_MIN_SOURCES = 1


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


class ResearchAgent:
    """Factory for a CrewAI Research Agent.

    The agent is responsible for:
    - Formulating targeted search queries from a broad topic
    - Retrieving web search results and scraping relevant pages
    - Extracting key points from each source
    - Assigning a relevance score (0–1) per source
    - Emitting a structured JSON block parseable by :class:`ResearchRunner`

    Example::

        tools = [SerperSearchTool(api_key="...")]
        agent = ResearchAgent.create(tools=tools)
    """

    @staticmethod
    def create(
        tools: list[BaseTool],
        model: str = "gpt-4o",
        verbose: bool = True,
        max_iter: int = 10,
    ) -> Agent:
        """Instantiate and return a configured CrewAI ``Agent``.

        Args:
            tools:    CrewAI-compatible tool instances.
            model:    OpenAI model identifier.
            verbose:  Log chain-of-thought.
            max_iter: Maximum reasoning iterations.

        Returns:
            A ready-to-use :class:`crewai.Agent`.
        """
        logger.info(
            "ResearchAgent.create — model=%s tools=%d", model, len(tools)
        )
        return Agent(
            role="Senior Research Specialist",
            goal=(
                "Conduct thorough, multi-angle research on the given topic. "
                "Gather information from diverse, credible sources. "
                "Extract key facts, statistics, expert opinions, and recent "
                "developments. Score each source for relevance (0.0–1.0). "
                "Return all findings in the required structured JSON format."
            ),
            backstory=(
                "You are an elite research specialist with 15+ years of experience "
                "in investigative and academic research. You excel at finding "
                "relevant sources, cross-referencing facts, and synthesising "
                "information from disparate fields. You always cite sources with "
                "URLs and flag uncertain information."
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


class ResearchRunner:
    """End-to-end orchestrator for a single research topic.

    Retry strategy (three levels):

    1. **HTTP level** — each search tool retries failed API calls up to
       ``tool_max_retries`` times with exponential back-off (handled inside
       the tool classes).
    2. **Run level** — if the agent raises an exception, the entire crew run
       is retried up to ``max_retries`` times.
    3. **Quality level** — if the parsed output has fewer than
       ``_MIN_SOURCES`` sources (empty result), the run is retried once with
       a broader query reformulation hint.

    Example::

        runner = ResearchRunner()
        output = runner.run("The impact of LLMs on scientific research")
        for src in output.top_sources:
            print(src.url, src.relevance_score)
    """

    def __init__(
        self,
        *,
        serper_api_key: str | None = None,
        tavily_api_key: str | None = None,
        model: str = "gpt-4o",
        max_retries: int = 3,
        tool_max_retries: int = 3,
        request_timeout: int = 30,
        verbose: bool = False,
    ) -> None:
        """Initialise the runner, preferring config from env when args omitted.

        Pass ``None`` (the default) for API keys to load them from the
        ``.env`` / environment.  Pass an explicit empty string ``""`` to
        deliberately use no key (forces DuckDuckGo fallback) — useful in
        tests that want to bypass config loading.

        Args:
            serper_api_key:   Serper API key. ``None`` → read from config.
            tavily_api_key:   Tavily API key. ``None`` → read from config.
            model:            OpenAI model for the agent.
            max_retries:      Run-level retry limit.
            tool_max_retries: HTTP-level retry limit inside search tools.
            request_timeout:  HTTP timeout in seconds.
            verbose:          Pass to CrewAI for chain-of-thought logging.
        """
        # Only consult config when keys are not explicitly supplied
        if serper_api_key is None or tavily_api_key is None:
            try:
                from src.config.config_loader import load_config

                cfg = load_config()
                if serper_api_key is None:
                    serper_api_key = cfg.serper_api_key.get_secret_value()
                if tavily_api_key is None:
                    tavily_api_key = cfg.tavily_api_key.get_secret_value()
                model = model or cfg.model_name
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not load config: %s — using defaults", exc)
                serper_api_key = serper_api_key or ""
                tavily_api_key = tavily_api_key or ""

        self._model = model
        self._max_retries = max_retries
        self._verbose = verbose

        self._search_tool, self._backend = self._select_backend(
            serper_api_key=serper_api_key,
            tavily_api_key=tavily_api_key,
            tool_max_retries=tool_max_retries,
            request_timeout=request_timeout,
        )
        logger.info(
            "ResearchRunner initialised — backend=%s model=%s",
            self._backend.value,
            self._model,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, topic: str) -> ResearchOutput:
        """Research *topic* and return a structured :class:`ResearchOutput`.

        Implements run-level retry: if the crew raises an exception or the
        result is empty, the run is retried up to ``max_retries`` times.

        Args:
            topic: Natural-language research topic or question.

        Returns:
            :class:`ResearchOutput` (``success=False`` on unrecoverable error).
        """
        logger.info("ResearchRunner.run — topic=%r", topic)
        start = time.perf_counter()

        last_error: str = ""
        for attempt in range(1, self._max_retries + 1):
            logger.info("Run attempt %d/%d", attempt, self._max_retries)
            try:
                raw_output, queries = self._execute_crew(topic, attempt)
                output = self._parse_output(
                    raw=raw_output,
                    topic=topic,
                    queries=queries,
                    duration=time.perf_counter() - start,
                    retry_count=attempt - 1,
                )
                if output.sources or attempt == self._max_retries:
                    output.success = bool(output.sources)
                    if not output.sources:
                        output.error = "No sources found after all retries."
                    logger.info(
                        "Run complete — success=%s sources=%d attempt=%d",
                        output.success,
                        len(output.sources),
                        attempt,
                    )
                    return output

                # Quality retry: got no sources, try again
                logger.warning(
                    "Attempt %d returned no sources, retrying…", attempt
                )
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                logger.error(
                    "Run attempt %d failed: %s", attempt, exc, exc_info=True
                )

        # All retries exhausted
        return ResearchOutput(
            topic=topic,
            success=False,
            error=last_error or "All retry attempts failed.",
            duration_secs=time.perf_counter() - start,
            retry_count=self._max_retries,
            backend=self._backend,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _select_backend(
        serper_api_key: str,
        tavily_api_key: str,
        tool_max_retries: int,
        request_timeout: int,
    ) -> tuple[BaseTool, SearchBackend]:
        """Pick the best available search tool and return it with its enum tag."""
        from src.tools.search_tools import (
            SerperSearchTool,
            TavilySearchTool,
            WebSearchTool,
        )

        placeholder_prefixes = ("your-", "tvly-...")

        def _is_real(key: str) -> bool:
            return bool(key) and not any(
                key.startswith(p) for p in placeholder_prefixes
            )

        if _is_real(serper_api_key):
            logger.info("Search backend: Serper")
            return (
                SerperSearchTool(
                    api_key=serper_api_key,
                    max_retries=tool_max_retries,
                    request_timeout=request_timeout,
                ),
                SearchBackend.SERPER,
            )
        if _is_real(tavily_api_key):
            logger.info("Search backend: Tavily")
            return (
                TavilySearchTool(
                    api_key=tavily_api_key,
                    max_retries=tool_max_retries,
                    request_timeout=request_timeout,
                ),
                SearchBackend.TAVILY,
            )

        logger.warning(
            "No search API key found — falling back to DuckDuckGo (free, limited)"
        )
        return WebSearchTool(max_retries=tool_max_retries), SearchBackend.DUCKDUCKGO

    def _execute_crew(
        self, topic: str, attempt: int
    ) -> tuple[str, list[str]]:
        """Build agent + task, kick off the crew, return (raw_output, queries)."""
        retry_hint = (
            "\n\nNote: previous attempt found no results. "
            "Try broader search queries this time."
            if attempt > 1
            else ""
        )

        queries: list[str] = []
        task_description = self._build_task_description(topic, retry_hint)

        agent = ResearchAgent.create(
            tools=[self._search_tool],
            model=self._model,
            verbose=self._verbose,
        )
        task = Task(
            description=task_description,
            expected_output=(
                f"A {_OUTPUT_FENCE_START} / {_OUTPUT_FENCE_END} JSON block "
                "containing sources, key_findings, and search_queries."
            ),
            agent=agent,
        )
        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=self._verbose,
        )

        logger.debug("Kicking off crew for topic=%r attempt=%d", topic, attempt)
        crew_output = crew.kickoff()

        raw: str
        if hasattr(crew_output, "raw"):
            raw = crew_output.raw
        else:
            raw = str(crew_output)

        return raw, queries

    @staticmethod
    def _build_task_description(topic: str, retry_hint: str = "") -> str:
        """Compose the task prompt that instructs the agent on output format."""
        return f"""Research the following topic thoroughly:

**Topic:** {topic}{retry_hint}

## Instructions

1. Generate 3–5 targeted search queries covering different angles of the topic.
2. Execute each query using the search tool.
3. For each result, extract 2–4 key bullet-point insights from the snippet.
4. Assign a relevance score (0.0–1.0) based on how directly the source addresses the topic.
5. Deduplicate results by URL.
6. Synthesise 5–8 overall key findings that span all sources.

## Required output format

You MUST include a JSON block delimited EXACTLY as shown below.
Do not omit or rename the delimiters.

{_OUTPUT_FENCE_START}
{{
  "search_queries": ["query 1", "query 2", "..."],
  "key_findings": [
    "Finding 1 — supported by source X",
    "Finding 2 — supported by source Y"
  ],
  "sources": [
    {{
      "url": "https://example.com/article",
      "title": "Article title",
      "snippet": "Short excerpt from the page",
      "key_points": ["Point A", "Point B"],
      "relevance_score": 0.92,
      "position": 1
    }}
  ]
}}
{_OUTPUT_FENCE_END}

You may include additional free-text analysis before or after the JSON block.
"""

    def _parse_output(
        self,
        raw: str,
        topic: str,
        queries: list[str],
        duration: float,
        retry_count: int,
    ) -> ResearchOutput:
        """Extract structured data from the agent's raw text output.

        Parsing strategy (in order):
        1. Extract JSON between ``RESEARCH_JSON_START`` … ``RESEARCH_JSON_END``.
        2. Fall back to the first ``{…}`` JSON blob in the output.
        3. Last resort: return a bare ``ResearchOutput`` with the raw text
           as the single key finding.
        """
        logger.debug("Parsing agent output (%d chars)", len(raw))

        data = self._extract_json(raw)
        if data is None:
            logger.warning("Could not parse JSON from agent output — using raw text")
            return ResearchOutput(
                topic=topic,
                key_findings=[raw[:2000]] if raw else [],
                search_queries=queries,
                duration_secs=duration,
                retry_count=retry_count,
                backend=self._backend,
            )

        sources: list[SourceResult] = []
        seen_urls: set[str] = set()
        for item in data.get("sources", []):
            url = item.get("url", "").strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            try:
                sources.append(
                    SourceResult(
                        url=url,
                        title=item.get("title", ""),
                        snippet=item.get("snippet", ""),
                        key_points=item.get("key_points", []),
                        relevance_score=item.get("relevance_score", 0.5),
                        position=item.get("position", 0),
                        backend=self._backend,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping malformed source %r: %s", url, exc)

        parsed_queries: list[str] = data.get("search_queries", queries)
        key_findings: list[str] = data.get("key_findings", [])

        logger.info(
            "Parsed output — sources=%d findings=%d queries=%d",
            len(sources),
            len(key_findings),
            len(parsed_queries),
        )

        return ResearchOutput(
            topic=topic,
            sources=sources,
            key_findings=key_findings,
            search_queries=parsed_queries,
            total_sources_examined=len(data.get("sources", [])),
            duration_secs=duration,
            retry_count=retry_count,
            backend=self._backend,
        )

    @staticmethod
    def _extract_json(text: str) -> dict | None:
        """Try to extract a JSON dict from *text* using two strategies."""
        # Strategy 1: delimited fence
        pattern = rf"{re.escape(_OUTPUT_FENCE_START)}\s*({{.*?}})\s*{re.escape(_OUTPUT_FENCE_END)}"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError as exc:
                logger.debug("Fenced JSON parse failed: %s", exc)

        # Strategy 2: largest {...} blob in the text
        candidates = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
        for candidate in sorted(candidates, key=len, reverse=True):
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue

        return None
