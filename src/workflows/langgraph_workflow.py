"""LangGraph workflow orchestrating the four-agent research pipeline.

Architecture
============
The pipeline models as a directed state graph with four typed nodes::

    START → research → summarize → write_report → quality_check
                                        ↑               │
                                        │  score < 70   │
                                        │  revisions ≤ 3│
                                        └───────────────┘
                                                        │
                                                       END (approved OR max revisions)

Features
========
- **Full state typing** — :class:`WorkflowState` TypedDict keeps every artefact
  produced by each node; reducers prevent accidental overwrites.
- **Revision loop** — conditional edge re-routes to the Report Writer when
  ``quality_score < quality_threshold`` (default 70), passing the QA feedback
  directly into the writer's prompt.
- **Maximum revision guard** — ``max_revisions`` (default 3) cap prevents
  infinite loops regardless of score.
- **SQLite-backed checkpointing** — every completed node writes its state to an
  SQLite database; interrupted runs can be resumed with the same ``thread_id``.
- **Execution trace** — every node appends a :class:`TraceEvent` to
  ``execution_trace``; the trace is also written to the Python logger.
- **Graph visualisation** — ``save_diagram()`` renders the compiled graph as a
  PNG using LangGraph's Mermaid renderer; falls back to saving a ``.mmd`` text
  file if the render fails (e.g. no network access).

Typical usage::

    from src.workflows.langgraph_workflow import ResearchGraph

    graph = ResearchGraph(quality_threshold=70, max_revisions=3)
    graph.save_diagram("workflow_diagram.png")

    final_state = graph.run("AI applications in drug discovery")
    print(final_state["report"])
    print(f"Quality: {final_state['quality_score']}/100")

    # Resume an interrupted run using the same thread_id:
    thread_id = "my-session-abc"
    state1 = graph.run("AI topic", thread_id=thread_id)
    # … process crashes …
    state2 = graph.run("AI topic", thread_id=thread_id)  # picks up where it left off
"""

from __future__ import annotations

import logging
import operator
import sqlite3
import time
import traceback
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from typing_extensions import Annotated, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from src.agents.quality_checker_agent import QualityCheckerRunner
from src.agents.report_writer_agent import ReportWriterAgent
from src.agents.research_agent import ResearchRunner
from src.agents.summarizer_agent import SummarizerAgent
from src.models.research_models import ResearchOutput, SourceResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_QUALITY_THRESHOLD = 70
_DEFAULT_MAX_REVISIONS = 3
_DEFAULT_DB_PATH = "workflow_state.db"
_DEFAULT_MODEL = "gpt-4o"


# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------


class TraceEvent(TypedDict):
    """A single entry in the workflow execution trace."""

    node: str
    started_at: str      # ISO-8601 UTC
    completed_at: str    # ISO-8601 UTC
    duration_secs: float
    status: str          # "success" | "error"
    details: dict        # node-specific metrics (scores, char counts, etc.)


class WorkflowState(TypedDict):
    """Complete mutable state shared across all LangGraph nodes.

    Fields produced by each stage are kept in clearly-named groups so that any
    node can inspect prior-stage outputs without coupling to internal runner
    details.

    The ``execution_trace`` field uses ``operator.add`` as its reducer so that
    each node *appends* its :class:`TraceEvent` rather than replacing the list.
    """

    # ── Input ────────────────────────────────────────────────────────────────
    topic: str

    # ── Research node ────────────────────────────────────────────────────────
    research_findings: list[str]        # key cross-source insights
    research_sources: list[dict]        # SourceResult.model_dump() list
    research_queries: list[str]         # queries the agent executed
    research_backend: str               # "serper" | "tavily" | "duckduckgo"
    research_success: bool

    # ── Summarize node ───────────────────────────────────────────────────────
    summary: str

    # ── Write Report node (updated each revision) ────────────────────────────
    report: str

    # ── Quality Check node (updated each revision) ──────────────────────────
    quality_score: int              # 0–100 aggregate
    quality_verdict: str            # "APPROVED" | "REVISION_REQUIRED"
    quality_revision_prompt: str    # targeted feedback for the writer
    quality_dimensions: list[dict]  # QualityDimension.model_dump() list
    quality_issues: list[dict]      # QualityIssue.model_dump() list
    quality_top_suggestions: list[str]

    # ── Control flow ─────────────────────────────────────────────────────────
    revision_count: int             # write_report → quality_check cycles done

    # ── Execution trace (append-only reducer) ────────────────────────────────
    execution_trace: Annotated[list[TraceEvent], operator.add]

    # ── Final status ─────────────────────────────────────────────────────────
    status: str     # "running" | "approved" | "max_revisions_reached" | "failed"
    error: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _trace_event(
    node: str,
    started_at: str,
    status: str,
    details: dict,
) -> TraceEvent:
    completed_at = _utc_now()
    # Calculate duration from ISO timestamps
    try:
        t0 = datetime.fromisoformat(started_at)
        t1 = datetime.fromisoformat(completed_at)
        duration = (t1 - t0).total_seconds()
    except Exception:
        duration = 0.0

    return TraceEvent(
        node=node,
        started_at=started_at,
        completed_at=completed_at,
        duration_secs=round(duration, 3),
        status=status,
        details=details,
    )


def _log_trace(event: TraceEvent) -> None:
    logger.info(
        "[Trace] node=%-18s status=%-8s duration=%.2fs | %s",
        event["node"],
        event["status"],
        event["duration_secs"],
        " ".join(f"{k}={v}" for k, v in event["details"].items()),
    )


def _research_output_to_state(output: ResearchOutput) -> dict[str, Any]:
    """Flatten a ResearchOutput into JSON-serialisable state fields."""
    return {
        "research_findings": output.key_findings,
        "research_sources": [s.model_dump() for s in output.sources],
        "research_queries": output.search_queries,
        "research_backend": output.backend.value,
        "research_success": output.success,
    }


def _sources_from_state(state: WorkflowState) -> list[SourceResult]:
    """Reconstruct SourceResult objects from the serialised state dicts."""
    results = []
    for d in state.get("research_sources", []):
        try:
            results.append(SourceResult(**d))
        except Exception:
            pass
    return results


def _build_research_context(state: WorkflowState, max_sources: int = 8) -> str:
    """Format research findings + top sources into a compact context block."""
    lines: list[str] = []
    if state.get("research_findings"):
        lines.append("## Key Research Findings\n")
        for i, f in enumerate(state["research_findings"][:8], 1):
            lines.append(f"{i}. {f}")
    sources = _sources_from_state(state)
    if sources:
        lines.append("\n## Top Sources\n")
        for src in sorted(sources, key=lambda s: s.relevance_score, reverse=True)[:max_sources]:
            lines.append(
                f"- **{src.title or src.url}** ({src.url})"
                f" — relevance {src.relevance_score:.2f}"
            )
            for pt in src.key_points[:2]:
                lines.append(f"  • {pt}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Node factories (closures over runner instances)
# ---------------------------------------------------------------------------


def _make_research_node(runner: ResearchRunner):
    """Return a research node bound to *runner*."""

    def research_node(state: WorkflowState) -> dict[str, Any]:
        node = "research"
        started_at = _utc_now()
        logger.info("[%s] Starting — topic=%r", node, state["topic"])

        try:
            output: ResearchOutput = runner.run(state["topic"])
            updates = _research_output_to_state(output)
            error_msg = output.error or ""
            status = "success" if output.success else "error"
            updates["status"] = "running"
            updates["error"] = error_msg

            event = _trace_event(
                node=node,
                started_at=started_at,
                status=status,
                details={
                    "sources": len(output.sources),
                    "findings": len(output.key_findings),
                    "backend": output.backend.value,
                    "retries": output.retry_count,
                },
            )
        except Exception as exc:
            logger.error("[%s] Failed: %s", node, exc, exc_info=True)
            updates = {
                "research_findings": [],
                "research_sources": [],
                "research_queries": [],
                "research_backend": "unknown",
                "research_success": False,
                "status": "failed",
                "error": str(exc),
            }
            event = _trace_event(
                node=node,
                started_at=started_at,
                status="error",
                details={"error": str(exc)[:200]},
            )

        _log_trace(event)
        updates["execution_trace"] = [event]
        return updates

    return research_node


def _make_summarize_node(model: str, verbose: bool):
    """Return a summarize node that runs a CrewAI SummarizerAgent."""

    def summarize_node(state: WorkflowState) -> dict[str, Any]:
        from crewai import Crew, Process, Task

        node = "summarize"
        started_at = _utc_now()
        logger.info("[%s] Starting", node)

        try:
            context = _build_research_context(state)
            task_desc = (
                f"Summarise the following research material on the topic "
                f"**{state['topic']}** into a concise, structured summary.\n\n"
                f"{context}\n\n"
                f"Requirements:\n"
                f"- 500–800 words\n"
                f"- Group insights by theme with clear sub-headings\n"
                f"- Highlight 5–8 most important findings as bullet points\n"
                f"- Note any significant gaps or uncertainties\n"
                f"- Preserve all source references that appear in the research"
            )
            agent = SummarizerAgent.create(model=model, verbose=verbose)
            task = Task(
                description=task_desc,
                expected_output=(
                    "A concise structured summary (500–800 words) with themed "
                    "sections and bullet-point key insights."
                ),
                agent=agent,
            )
            crew = Crew(
                agents=[agent],
                tasks=[task],
                process=Process.sequential,
                verbose=verbose,
            )
            crew_output = crew.kickoff()
            summary: str = (
                crew_output.raw
                if hasattr(crew_output, "raw")
                else str(crew_output)
            )

            event = _trace_event(
                node=node,
                started_at=started_at,
                status="success",
                details={"chars": len(summary)},
            )
            updates: dict[str, Any] = {
                "summary": summary,
                "status": "running",
                "error": "",
            }
        except Exception as exc:
            logger.error("[%s] Failed: %s", node, exc, exc_info=True)
            event = _trace_event(
                node=node,
                started_at=started_at,
                status="error",
                details={"error": str(exc)[:200]},
            )
            updates = {
                "summary": "",
                "status": "failed",
                "error": str(exc),
            }

        _log_trace(event)
        updates["execution_trace"] = [event]
        return updates

    return summarize_node


def _make_write_report_node(model: str, verbose: bool):
    """Return a write-report node that runs a CrewAI ReportWriterAgent.

    On revision rounds (``revision_count > 0``) the node receives the previous
    report and the quality feedback in the task prompt, so the agent can make
    targeted improvements.
    """

    def write_report_node(state: WorkflowState) -> dict[str, Any]:
        from crewai import Crew, Process, Task

        node = "write_report"
        revision = state.get("revision_count", 0)
        started_at = _utc_now()
        logger.info("[%s] Starting (revision=%d)", node, revision)

        try:
            context = _build_research_context(state)

            if revision == 0:
                # ── First pass: write from scratch ──────────────────────────
                task_desc = (
                    f"Write a comprehensive, professional research report on the "
                    f"topic: **{state['topic']}**\n\n"
                    f"Use this research summary as your primary source:\n"
                    f"{state.get('summary', '(no summary yet)')}\n\n"
                    f"And this research evidence:\n{context}\n\n"
                    f"## Required report structure (Markdown)\n"
                    f"# Research Report: {state['topic']}\n"
                    f"**Date:** {datetime.now().strftime('%Y-%m-%d')}  "
                    f"|  **System:** Multi-Agent Research Pipeline\n\n"
                    f"## Executive Summary\n"
                    f"## Background & Context\n"
                    f"## Key Findings\n"
                    f"## Analysis & Implications\n"
                    f"## Conclusions\n"
                    f"## Recommendations\n"
                    f"## References\n\n"
                    f"Guidelines: professional tone, 800–1 500 words, "
                    f"every claim supported by the research evidence."
                )
            else:
                # ── Revision pass: targeted improvements ────────────────────
                feedback = state.get("quality_revision_prompt") or (
                    "\n".join(
                        f"- {s}"
                        for s in state.get("quality_top_suggestions", [])[:5]
                    )
                )
                task_desc = (
                    f"Revise the research report below based on the quality "
                    f"feedback provided. This is revision {revision} of "
                    f"max {_DEFAULT_MAX_REVISIONS}.\n\n"
                    f"## Quality Feedback (score was {state.get('quality_score', 0)}/100)\n"
                    f"{feedback}\n\n"
                    f"## Current Report (revise this)\n"
                    f"{state.get('report', '')}\n\n"
                    f"## Additional Research Context\n"
                    f"{context}\n\n"
                    f"Return the COMPLETE revised report in Markdown, preserving "
                    f"the original section structure unless the feedback demands changes."
                )

            agent = ReportWriterAgent.create(model=model, verbose=verbose)
            task = Task(
                description=task_desc,
                expected_output=(
                    "A polished Markdown research report with all required sections, "
                    "well-evidenced claims, and professional prose."
                ),
                agent=agent,
            )
            crew = Crew(
                agents=[agent],
                tasks=[task],
                process=Process.sequential,
                verbose=verbose,
            )
            crew_output = crew.kickoff()
            report: str = (
                crew_output.raw
                if hasattr(crew_output, "raw")
                else str(crew_output)
            )

            event = _trace_event(
                node=node,
                started_at=started_at,
                status="success",
                details={"chars": len(report), "revision": revision},
            )
            updates: dict[str, Any] = {
                "report": report,
                "status": "running",
                "error": "",
            }
        except Exception as exc:
            logger.error("[%s] Failed: %s", node, exc, exc_info=True)
            event = _trace_event(
                node=node,
                started_at=started_at,
                status="error",
                details={"error": str(exc)[:200], "revision": revision},
            )
            updates = {
                "report": state.get("report", ""),
                "status": "failed",
                "error": str(exc),
            }

        _log_trace(event)
        updates["execution_trace"] = [event]
        return updates

    return write_report_node


def _make_quality_check_node(runner: QualityCheckerRunner):
    """Return a quality-check node bound to *runner*."""

    def quality_check_node(state: WorkflowState) -> dict[str, Any]:
        node = "quality_check"
        revision = state.get("revision_count", 0)
        started_at = _utc_now()
        logger.info("[%s] Starting (revision=%d)", node, revision)

        try:
            # Reconstruct a lightweight ResearchOutput for cross-checking
            sources = _sources_from_state(state)
            from src.models.research_models import SearchBackend

            research_output = ResearchOutput(
                topic=state["topic"],
                sources=sources,
                key_findings=state.get("research_findings", []),
                search_queries=state.get("research_queries", []),
                success=state.get("research_success", False),
                backend=SearchBackend(state.get("research_backend", "unknown")),
            )

            qr = runner.check(
                report_content=state.get("report", ""),
                research_output=research_output,
                topic=state["topic"],
            )

            verdict = qr.verdict
            next_revision = revision + 1
            final_status: str
            if verdict == "APPROVED":
                final_status = "approved"
                logger.info(
                    "[%s] Report APPROVED — score=%d/100 after %d revision(s)",
                    node,
                    qr.overall_score,
                    next_revision,
                )
            else:
                final_status = "running"  # may loop or end at max revisions
                logger.info(
                    "[%s] REVISION REQUIRED — score=%d/100 revision=%d",
                    node,
                    qr.overall_score,
                    next_revision,
                )

            event = _trace_event(
                node=node,
                started_at=started_at,
                status="success",
                details={
                    "score": qr.overall_score,
                    "verdict": verdict,
                    "issues": len(qr.issues),
                    "revision": next_revision,
                },
            )
            updates: dict[str, Any] = {
                "quality_score": qr.overall_score,
                "quality_verdict": verdict,
                "quality_revision_prompt": qr.revision_prompt or "",
                "quality_dimensions": [d.model_dump() for d in qr.dimensions],
                "quality_issues": [i.model_dump() for i in qr.issues],
                "quality_top_suggestions": qr.top_suggestions,
                "revision_count": next_revision,
                "status": final_status,
                "error": "",
            }
        except Exception as exc:
            logger.error("[%s] Failed: %s", node, exc, exc_info=True)
            event = _trace_event(
                node=node,
                started_at=started_at,
                status="error",
                details={"error": str(exc)[:200]},
            )
            updates = {
                "quality_score": 0,
                "quality_verdict": "REVISION_REQUIRED",
                "quality_revision_prompt": "",
                "quality_dimensions": [],
                "quality_issues": [],
                "quality_top_suggestions": [],
                "revision_count": state.get("revision_count", 0) + 1,
                "status": "failed",
                "error": str(exc),
            }

        _log_trace(event)
        updates["execution_trace"] = [event]
        return updates

    return quality_check_node


# ---------------------------------------------------------------------------
# Conditional edge routing
# ---------------------------------------------------------------------------


def _make_router(quality_threshold: int, max_revisions: int):
    """Return the conditional-edge function for the quality_check node.

    Decision matrix:

    +-----------------------+-------------------+---------------------------+
    | Condition             | revision_count    | Route                     |
    +-----------------------+-------------------+---------------------------+
    | score >= threshold    | any               | END (approved)            |
    | score <  threshold    | < max_revisions   | write_report (revise)     |
    | score <  threshold    | >= max_revisions  | END (max revisions hit)   |
    +-----------------------+-------------------+---------------------------+
    """

    def route_after_quality_check(
        state: WorkflowState,
    ) -> Literal["write_report", "__end__"]:
        score = state.get("quality_score", 0)
        verdict = state.get("quality_verdict", "REVISION_REQUIRED")
        revisions = state.get("revision_count", 0)

        if verdict == "APPROVED" or score >= quality_threshold:
            logger.info(
                "[Router] → END (APPROVED — score=%d revisions=%d)",
                score, revisions,
            )
            return "__end__"

        if revisions >= max_revisions:
            logger.warning(
                "[Router] → END (max revisions=%d reached — score=%d)",
                max_revisions, score,
            )
            return "__end__"

        logger.info(
            "[Router] → write_report (score=%d < %d, revision %d/%d)",
            score, quality_threshold, revisions, max_revisions,
        )
        return "write_report"

    return route_after_quality_check


# ---------------------------------------------------------------------------
# Graph class
# ---------------------------------------------------------------------------


class ResearchGraph:
    """LangGraph state machine orchestrating the four-agent research pipeline.

    Args:
        model:              OpenAI model identifier for all agents.
        quality_threshold:  Minimum score (0–100) to approve a report.
        max_revisions:      Maximum write_report→quality_check cycles.
        db_path:            Path for the SQLite checkpoint database.
                            Pass ``None`` to use in-memory checkpointing only.
        verbose:            Forward verbose flag to CrewAI agents.

    Example::

        graph = ResearchGraph(
            quality_threshold=70,
            max_revisions=3,
            db_path="my_research.db",
        )
        graph.save_diagram("workflow_diagram.png")
        final = graph.run("The future of nuclear fusion energy")
        print(final["report"])
    """

    def __init__(
        self,
        *,
        model: str = _DEFAULT_MODEL,
        quality_threshold: int = _DEFAULT_QUALITY_THRESHOLD,
        max_revisions: int = _DEFAULT_MAX_REVISIONS,
        db_path: str | None = _DEFAULT_DB_PATH,
        verbose: bool = False,
    ) -> None:
        if not 0 <= quality_threshold <= 100:
            raise ValueError(f"quality_threshold must be 0–100, got {quality_threshold}")
        if max_revisions < 1:
            raise ValueError(f"max_revisions must be ≥ 1, got {max_revisions}")

        self._model = model
        self._quality_threshold = quality_threshold
        self._max_revisions = max_revisions
        self._db_path = db_path
        self._verbose = verbose

        # Instantiate shared runners (created once, reused across runs)
        self._research_runner = ResearchRunner(model=model, verbose=verbose)
        self._qc_runner = QualityCheckerRunner(
            model=model,
            quality_threshold=quality_threshold,
            verbose=verbose,
        )

        # Build and compile the graph
        self._checkpointer = self._build_checkpointer()
        self._compiled = self._build_graph()

        logger.info(
            "ResearchGraph ready — model=%s threshold=%d max_revisions=%d db=%s",
            model, quality_threshold, max_revisions, db_path or "memory",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        topic: str,
        thread_id: str | None = None,
    ) -> WorkflowState:
        """Run the full pipeline for *topic* and return the final state.

        Passing the same *thread_id* to a subsequent call resumes from the
        last successful checkpoint (SQLite persistence required).

        Args:
            topic:     Research topic or question.
            thread_id: Checkpoint key.  Auto-generated when ``None``.

        Returns:
            Final :class:`WorkflowState` after all nodes complete.
        """
        thread_id = thread_id or str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        initial = self._initial_state(topic)

        logger.info(
            "ResearchGraph.run — topic=%r thread_id=%s", topic, thread_id
        )
        start = time.perf_counter()
        final_state: WorkflowState = self._compiled.invoke(initial, config=config)
        duration = time.perf_counter() - start

        self._log_summary(final_state, duration)
        return final_state

    def stream(
        self,
        topic: str,
        thread_id: str | None = None,
    ):
        """Stream intermediate states as each node completes.

        Yields ``(node_name, partial_state)`` tuples in execution order.

        Args:
            topic:     Research topic or question.
            thread_id: Checkpoint key.  Auto-generated when ``None``.

        Yields:
            ``(str, dict)`` — node name and its output state update.
        """
        thread_id = thread_id or str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        initial = self._initial_state(topic)

        logger.info(
            "ResearchGraph.stream — topic=%r thread_id=%s", topic, thread_id
        )
        for step in self._compiled.stream(initial, config=config):
            for node_name, partial_state in step.items():
                logger.debug("[Stream] Completed node: %s", node_name)
                yield node_name, partial_state

    def save_diagram(self, path: str = "workflow_diagram.png") -> Path:
        """Render the compiled graph and save it to *path*.

        Primary renderer: LangGraph's ``draw_mermaid_png()`` (uses the free
        mermaid.ink online API — requires internet access).

        Fallback: saves a ``*.mmd`` Mermaid source file next to *path* if the
        PNG render fails.

        Args:
            path: Destination file path for the PNG.

        Returns:
            The path that was actually written (PNG or .mmd fallback).
        """
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        graph_repr = self._compiled.get_graph()

        # ── Try Mermaid PNG via mermaid.ink ──────────────────────────────────
        try:
            png_bytes: bytes = graph_repr.draw_mermaid_png()
            dest.write_bytes(png_bytes)
            logger.info("Workflow diagram saved → %s (%d bytes)", dest, len(png_bytes))
            return dest
        except Exception as png_exc:
            logger.warning(
                "PNG render failed (%s) — saving Mermaid source instead.", png_exc
            )

        # ── Fallback: save raw Mermaid source ────────────────────────────────
        mmd_path = dest.with_suffix(".mmd")
        try:
            mermaid_text: str = graph_repr.draw_mermaid()
            mmd_path.write_text(mermaid_text, encoding="utf-8")
            logger.info("Mermaid source saved → %s", mmd_path)
            return mmd_path
        except Exception as mmd_exc:
            logger.error("Mermaid source export also failed: %s", mmd_exc)
            raise RuntimeError(
                f"Could not render workflow diagram: PNG error={png_exc}; "
                f"Mermaid error={mmd_exc}"
            ) from mmd_exc

    def print_trace(self, state: WorkflowState) -> None:
        """Pretty-print the execution trace from a finished *state* to stdout."""
        trace = state.get("execution_trace", [])
        if not trace:
            print("No execution trace available.")
            return

        print("\n" + "=" * 60)
        print(" EXECUTION TRACE")
        print("=" * 60)
        total = 0.0
        for ev in trace:
            status_icon = "✓" if ev["status"] == "success" else "✗"
            print(
                f"  {status_icon} {ev['node']:<18}  {ev['duration_secs']:6.2f}s"
                f"  [{ev['status']}]"
            )
            for k, v in ev["details"].items():
                print(f"       {k}: {v}")
            total += ev["duration_secs"]

        print("-" * 60)
        print(f"  Total wall-clock (sum of nodes): {total:.2f}s")
        print(
            f"  Final score : {state.get('quality_score', 0)}/100"
            f"  verdict: {state.get('quality_verdict', 'N/A')}"
        )
        print(f"  Revisions   : {state.get('revision_count', 0)}")
        print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_checkpointer(self):
        """Build the appropriate checkpointer based on ``db_path``."""
        if self._db_path is None:
            logger.info("Using in-memory checkpointer (no persistence)")
            return MemorySaver()

        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        logger.info("SQLite checkpointer initialised → %s", self._db_path)
        return checkpointer

    def _build_graph(self):
        """Construct and compile the LangGraph StateGraph."""
        builder: StateGraph = StateGraph(WorkflowState)

        # ── Register nodes ────────────────────────────────────────────────────
        builder.add_node("research", _make_research_node(self._research_runner))
        builder.add_node("summarize", _make_summarize_node(self._model, self._verbose))
        builder.add_node("write_report", _make_write_report_node(self._model, self._verbose))
        builder.add_node("quality_check", _make_quality_check_node(self._qc_runner))

        # ── Static edges ──────────────────────────────────────────────────────
        builder.add_edge(START, "research")
        builder.add_edge("research", "summarize")
        builder.add_edge("summarize", "write_report")
        builder.add_edge("write_report", "quality_check")

        # ── Conditional edge: approve ↔ revise ────────────────────────────────
        router = _make_router(self._quality_threshold, self._max_revisions)
        builder.add_conditional_edges(
            "quality_check",
            router,
            {
                "write_report": "write_report",
                "__end__": END,
            },
        )

        logger.info("ResearchGraph compiled (threshold=%d, max_revisions=%d)",
                    self._quality_threshold, self._max_revisions)
        return builder.compile(checkpointer=self._checkpointer)

    @staticmethod
    def _initial_state(topic: str) -> WorkflowState:
        """Return a zeroed-out :class:`WorkflowState` for *topic*."""
        return WorkflowState(
            topic=topic,
            research_findings=[],
            research_sources=[],
            research_queries=[],
            research_backend="unknown",
            research_success=False,
            summary="",
            report="",
            quality_score=0,
            quality_verdict="REVISION_REQUIRED",
            quality_revision_prompt="",
            quality_dimensions=[],
            quality_issues=[],
            quality_top_suggestions=[],
            revision_count=0,
            execution_trace=[],
            status="running",
            error="",
        )

    @staticmethod
    def _log_summary(state: WorkflowState, duration: float) -> None:
        """Log the pipeline summary after a completed run."""
        trace = state.get("execution_trace", [])
        node_summary = ", ".join(
            f"{ev['node']}({ev['status'][0].upper()})" for ev in trace
        )
        logger.info(
            "Pipeline finished in %.1fs — score=%d/100 verdict=%s revisions=%d | %s",
            duration,
            state.get("quality_score", 0),
            state.get("quality_verdict", "N/A"),
            state.get("revision_count", 0),
            node_summary,
        )
