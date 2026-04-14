"""Report Writer Agent — turns summaries into polished research reports."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from crewai import Agent

if TYPE_CHECKING:
    from crewai.tools import BaseTool

logger = logging.getLogger(__name__)


class ReportWriterAgent:
    """Factory for a CrewAI Report Writer Agent.

    Responsibilities:
    - Receive the distilled summaries from the Summarizer Agent
    - Structure the material into a professional research report
    - Apply consistent formatting: executive summary, body sections,
      conclusions, and references
    - Adapt tone and depth to the intended audience

    Example::

        agent = ReportWriterAgent.create(tools=[FileWriterTool()])
    """

    @staticmethod
    def create(
        tools: list[BaseTool] | None = None,
        model: str = "gpt-4o",
        verbose: bool = True,
        max_iter: int = 7,
    ) -> Agent:
        """Instantiate and return a configured CrewAI ``Agent``.

        Args:
            tools:    Optional tools (e.g. file writer to persist the report).
            model:    OpenAI model identifier.
            verbose:  Whether the agent logs its chain-of-thought.
            max_iter: Maximum reasoning iterations.

        Returns:
            A ready-to-use :class:`crewai.Agent`.
        """
        tools = tools or []
        logger.info("Creating ReportWriterAgent with model=%s", model)

        return Agent(
            role="Senior Technical Report Writer",
            goal=(
                "Produce well-structured, professional research reports from the "
                "provided summaries and research findings. "
                "Reports must include: an executive summary, clearly organised body "
                "sections with headings, data-backed conclusions, and a reference list. "
                "The writing should be clear, authoritative, and free of jargon unless "
                "domain-appropriate."
            ),
            backstory=(
                "You are a veteran technical writer with a background in both "
                "journalism and academic publishing. You have produced hundreds of "
                "research reports for Fortune 500 companies, government agencies, and "
                "peer-reviewed journals. You know exactly how to structure information "
                "for maximum clarity and impact, and you take pride in flawless "
                "grammar and consistent style."
            ),
            tools=tools,
            llm=model,
            verbose=verbose,
            max_iter=max_iter,
            allow_delegation=False,
            memory=True,
        )
