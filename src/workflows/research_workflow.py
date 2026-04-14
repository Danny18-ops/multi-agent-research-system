"""CrewAI-based research workflow that orchestrates all four agents."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from crewai import Crew, Process, Task

from src.agents import (
    QualityCheckerAgent,
    ReportWriterAgent,
    ResearchAgent,
    SummarizerAgent,
)
from src.config import get_settings
from src.tools import (
    FileReaderTool,
    FileWriterTool,
    TavilySearchTool,
    WebScraperTool,
    WebSearchTool,
)

logger = logging.getLogger(__name__)


@dataclass
class ResearchResult:
    """Container returned by :class:`ResearchWorkflow` after a run.

    Attributes:
        topic:          The original research topic.
        raw_research:   Notes produced by the Research Agent.
        summary:        Condensed insights from the Summarizer Agent.
        report:         Full structured report from the Report Writer Agent.
        quality_review: Feedback and score from the Quality Checker Agent.
        output_path:    Path where the final report was saved (if any).
        success:        ``True`` when the report passed quality checks.
        duration_secs:  Total wall-clock time for the run.
    """

    topic: str
    raw_research: str = ""
    summary: str = ""
    report: str = ""
    quality_review: str = ""
    output_path: Path | None = None
    success: bool = False
    duration_secs: float = 0.0
    metadata: dict = field(default_factory=dict)


class ResearchWorkflow:
    """End-to-end research pipeline powered by CrewAI.

    The pipeline executes four sequential tasks:

    1. **Research**       — web search + page scraping
    2. **Summarise**      — extract key insights from raw notes
    3. **Write report**   — produce a structured Markdown report
    4. **Quality check**  — validate and score the report

    Example::

        workflow = ResearchWorkflow()
        result = workflow.run("The impact of AI on healthcare in 2024")
        print(result.report)
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._build_tools()
        self._build_agents()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _build_tools(self) -> None:
        """Instantiate tools, preferring Tavily when an API key is set."""
        search_tool: WebSearchTool | TavilySearchTool
        if self._settings.tavily_api_key:
            search_tool = TavilySearchTool(api_key=self._settings.tavily_api_key)
            logger.info("Using Tavily search backend")
        else:
            search_tool = WebSearchTool()
            logger.info("Using DuckDuckGo search backend (no Tavily key)")

        self._search_tool = search_tool
        self._scraper_tool = WebScraperTool(request_timeout=self._settings.request_timeout)
        self._file_writer = FileWriterTool()
        self._file_reader = FileReaderTool()

    def _build_agents(self) -> None:
        """Instantiate all four specialist agents."""
        model = self._settings.openai_model
        self._researcher = ResearchAgent.create(
            tools=[self._search_tool, self._scraper_tool],
            model=model,
        )
        self._summarizer = SummarizerAgent.create(
            tools=[self._file_reader],
            model=model,
        )
        self._writer = ReportWriterAgent.create(
            tools=[self._file_writer],
            model=model,
        )
        self._checker = QualityCheckerAgent.create(
            tools=[self._search_tool, self._file_reader],
            model=model,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, topic: str) -> ResearchResult:
        """Execute the full research pipeline for *topic*.

        Args:
            topic: A natural-language description of what to research.

        Returns:
            A :class:`ResearchResult` with all intermediate and final outputs.
        """
        logger.info("Starting research pipeline for topic: %s", topic)
        start_time = datetime.now()
        result = ResearchResult(topic=topic)

        tasks = self._build_tasks(topic)
        crew = Crew(
            agents=[self._researcher, self._summarizer, self._writer, self._checker],
            tasks=tasks,
            process=Process.sequential,
            verbose=True,
            memory=True,
            max_rpm=10,  # rate-limit OpenAI calls
        )

        try:
            crew_output = crew.kickoff()
            result = self._parse_output(crew_output, result)
            result.success = True
            logger.info("Research pipeline completed successfully")
        except Exception as exc:  # noqa: BLE001
            logger.error("Research pipeline failed: %s", exc, exc_info=True)
            result.metadata["error"] = str(exc)

        result.duration_secs = (datetime.now() - start_time).total_seconds()
        logger.info("Pipeline finished in %.1f seconds", result.duration_secs)
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_tasks(self, topic: str) -> list[Task]:
        """Build the four sequential CrewAI Tasks for *topic*."""
        max_results = self._settings.max_search_results
        output_dir = self._settings.output_dir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        research_task = Task(
            description=(
                f"Research the following topic thoroughly:\n\n**Topic:** {topic}\n\n"
                f"Steps:\n"
                f"1. Generate 3-5 targeted search queries that cover different angles.\n"
                f"2. Execute each query using the search tool (max {max_results} results each).\n"
                f"3. Scrape and read the most relevant pages.\n"
                f"4. Compile comprehensive notes covering:\n"
                f"   - Key facts and statistics\n"
                f"   - Expert opinions and quotes\n"
                f"   - Recent developments (last 12 months if relevant)\n"
                f"   - Contrasting viewpoints\n"
                f"   - Source URLs for each piece of information\n"
                f"5. Save the notes to 'research_notes_{timestamp}.txt'."
            ),
            expected_output=(
                "Detailed research notes (1500–3000 words) covering the topic from "
                "multiple angles, with source URLs cited inline."
            ),
            agent=self._researcher,
        )

        summarize_task = Task(
            description=(
                "Analyse the research notes from the previous task and produce a "
                "structured summary.\n\n"
                "Requirements:\n"
                "- Extract the 5-10 most important insights\n"
                "- Identify key themes and patterns\n"
                "- Highlight any surprising or counterintuitive findings\n"
                "- Note any significant gaps or uncertainties in the research\n"
                "- Keep the summary to 500-800 words\n"
                "- Use bullet points and short sections for readability"
            ),
            expected_output=(
                "A concise, well-structured summary (500-800 words) with key insights "
                "grouped by theme. Bullet-point format preferred."
            ),
            agent=self._summarizer,
            context=[research_task],
        )

        write_report_task = Task(
            description=(
                f"Write a comprehensive research report on '{topic}' using the "
                "research notes and summary provided.\n\n"
                "Report structure (Markdown):\n"
                "# Research Report: [Topic]\n"
                "**Date:** [today's date]  |  **Prepared by:** AI Research System\n\n"
                "## Executive Summary\n"
                "## Background & Context\n"
                "## Key Findings\n"
                "### [Finding 1 heading]\n"
                "### [Finding 2 heading]\n"
                "### [Finding 3 heading]\n"
                "## Analysis & Implications\n"
                "## Conclusions\n"
                "## Recommendations\n"
                "## References\n\n"
                "Guidelines:\n"
                "- Professional, authoritative tone\n"
                "- Support every claim with evidence from the research\n"
                "- Minimum 800 words, maximum 2000 words\n"
                f"- Save the report as 'report_{timestamp}.md' in '{output_dir}'"
            ),
            expected_output=(
                "A polished, professional Markdown research report saved to disk, "
                "with all sections filled in and references listed."
            ),
            agent=self._writer,
            context=[research_task, summarize_task],
        )

        quality_check_task = Task(
            description=(
                "Review the research report produced by the Report Writer.\n\n"
                "Evaluate on these dimensions (score each 0–10):\n"
                "1. **Accuracy** — Are all claims supported by the research?\n"
                "2. **Completeness** — Does the report cover the topic adequately?\n"
                "3. **Clarity** — Is the writing clear and well-structured?\n"
                "4. **Evidence quality** — Are sources credible and properly cited?\n"
                "5. **Logical consistency** — Do conclusions follow from findings?\n\n"
                "Output format:\n"
                "- Dimension scores (table)\n"
                "- Overall quality score: X/10  (normalised to 0.0–1.0)\n"
                "- Strengths (bullet list)\n"
                "- Issues found (bullet list with severity: HIGH / MEDIUM / LOW)\n"
                "- Verdict: APPROVED or REVISION REQUIRED\n"
                "- If REVISION REQUIRED, list specific changes needed"
            ),
            expected_output=(
                "A structured quality review with individual dimension scores, "
                "overall score, strengths, issues, and a clear APPROVED / "
                "REVISION REQUIRED verdict."
            ),
            agent=self._checker,
            context=[research_task, summarize_task, write_report_task],
        )

        return [research_task, summarize_task, write_report_task, quality_check_task]

    @staticmethod
    def _parse_output(crew_output: object, result: ResearchResult) -> ResearchResult:
        """Populate *result* from the crew's raw output object."""
        # CrewAI >= 0.80 returns a CrewOutput object; fall back gracefully
        if hasattr(crew_output, "tasks_output") and crew_output.tasks_output:
            outputs = [t.raw for t in crew_output.tasks_output]
            if len(outputs) >= 1:
                result.raw_research = outputs[0]
            if len(outputs) >= 2:
                result.summary = outputs[1]
            if len(outputs) >= 3:
                result.report = outputs[2]
            if len(outputs) >= 4:
                result.quality_review = outputs[3]
        elif hasattr(crew_output, "raw"):
            result.report = crew_output.raw
        else:
            result.report = str(crew_output)

        return result
