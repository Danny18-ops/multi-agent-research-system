"""Summarizer Agent — distils raw research into concise key insights."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from crewai import Agent

if TYPE_CHECKING:
    from crewai.tools import BaseTool

logger = logging.getLogger(__name__)


class SummarizerAgent:
    """Factory for a CrewAI Summarizer Agent.

    Responsibilities:
    - Read and analyse the raw notes produced by the Research Agent
    - Identify the most important facts, trends, and insights
    - Eliminate redundancy and noise
    - Produce a structured, digestible summary ready for the Report Writer

    Example::

        agent = SummarizerAgent.create(tools=[])
    """

    @staticmethod
    def create(
        tools: list[BaseTool] | None = None,
        model: str = "gpt-4o",
        verbose: bool = True,
        max_iter: int = 5,
    ) -> Agent:
        """Instantiate and return a configured CrewAI ``Agent``.

        Args:
            tools:    Optional tools (e.g. file reader for loading saved notes).
            model:    OpenAI model identifier.
            verbose:  Whether the agent logs its chain-of-thought.
            max_iter: Maximum reasoning iterations.

        Returns:
            A ready-to-use :class:`crewai.Agent`.
        """
        tools = tools or []
        logger.info("Creating SummarizerAgent with model=%s", model)

        return Agent(
            role="Expert Information Synthesiser",
            goal=(
                "Transform large volumes of raw research material into clear, "
                "concise summaries that capture the most important insights. "
                "Preserve nuance and accuracy while eliminating redundancy. "
                "Structure findings by theme or importance."
            ),
            backstory=(
                "You are a world-class analyst who has summarised thousands of "
                "research papers, news articles, and technical documents. "
                "You have a gift for spotting the signal in the noise and for "
                "expressing complex ideas in plain language without losing depth. "
                "You always maintain the original meaning and never introduce "
                "information not present in the source material."
            ),
            tools=tools,
            llm=model,
            verbose=verbose,
            max_iter=max_iter,
            allow_delegation=False,
            memory=True,
        )
