#!/usr/bin/env python3
"""Multi-Agent Autonomous Research System — CLI entry point.

Orchestrates four AI agents (Research → Summarize → Write Report → Quality Check)
through a LangGraph state machine, displaying live progress and saving the final
report to disk.

Usage examples::

    # Basic run (LangGraph backend, streams progress)
    python main.py "The impact of multi-agent AI on scientific research"

    # CrewAI backend (no streaming, single crew)
    python main.py "Quantum computing in 2025" --backend crewai

    # Adjust quality threshold and revision cap
    python main.py "Fusion energy" --quality-threshold 75 --max-revisions 2

    # Save run metadata JSON alongside the report
    python main.py "Climate tech" --save-metadata

    # Verbose debug logging
    python main.py "AI governance" --log-level DEBUG
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

# ── Rich — all console output goes through here ───────────────────────────────
from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

console = Console(highlight=False)
err_console = Console(stderr=True)

# ── Logging ────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Pipeline stage definitions ─────────────────────────────────────────────────
_STAGES = [
    {
        "node": "research",
        "label": "Research Agent",
        "icon": "🔍",
        "desc": "Searching the web and gathering sources",
    },
    {
        "node": "summarize",
        "label": "Summarizer Agent",
        "icon": "📝",
        "desc": "Distilling key insights from raw notes",
    },
    {
        "node": "write_report",
        "label": "Report Writer Agent",
        "icon": "✍️",
        "desc": "Composing the structured Markdown report",
    },
    {
        "node": "quality_check",
        "label": "Quality Checker Agent",
        "icon": "🔬",
        "desc": "Evaluating report quality (0–100)",
    },
]

_NODE_TO_IDX = {s["node"]: i for i, s in enumerate(_STAGES)}


# ─────────────────────────────────────────────────────────────────────────────
# Token estimation
# ─────────────────────────────────────────────────────────────────────────────


def _count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Estimate token count for *text* using tiktoken.

    Falls back to ``len(text) // 4`` if tiktoken is unavailable or the model
    is not recognised.
    """
    try:
        import tiktoken

        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception:
        return max(0, len(text) // 4)


def _estimate_stage_tokens(state: dict[str, Any], model: str) -> dict[str, dict[str, int]]:
    """Estimate per-stage output token counts from the final workflow state.

    Input tokens are approximated as ``output × 2`` (prompts are typically
    longer than the model's reply).  These are *estimates* — exact counts
    require intercepting the underlying LLM calls.

    Returns a dict keyed by stage label with ``input`` and ``output`` sub-keys.
    """
    research_text = "\n".join(state.get("research_findings", []))
    for src in state.get("research_sources", []):
        research_text += f"\n{src.get('title', '')} {src.get('snippet', '')}"

    summary_text = state.get("summary", "")
    report_text = state.get("report", "")
    quality_text = " ".join(state.get("quality_top_suggestions", []))
    for dim in state.get("quality_dimensions", []):
        quality_text += f" {dim.get('name', '')} score={dim.get('score', 0)}"

    stage_texts = {
        "Research Agent": research_text,
        "Summarizer Agent": summary_text,
        "Report Writer Agent": report_text,
        "Quality Checker Agent": quality_text,
    }
    result: dict[str, dict[str, int]] = {}
    for label, text in stage_texts.items():
        out_tok = _count_tokens(text, model)
        result[label] = {"output": out_tok, "input": out_tok * 2}
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Rich progress display helpers
# ─────────────────────────────────────────────────────────────────────────────

_STATUS_WAITING = Text("Waiting", style="dim")
_STATUS_DONE_OK = Text("✓  Done", style="bold green")
_STATUS_DONE_ERR = Text("✗  Error", style="bold red")


def _build_progress_table(
    stage_states: list[dict[str, Any]],
    spinner_frame: str = "⠋",
) -> Table:
    """Render the pipeline progress as a Rich Table.

    Args:
        stage_states: One dict per stage with keys: status, duration, details.
        spinner_frame: Current spinner character for the active stage.
    """
    table = Table(
        show_header=True,
        header_style="bold cyan",
        box=None,
        padding=(0, 1),
        expand=True,
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("Agent", min_width=22)
    table.add_column("Status", min_width=14)
    table.add_column("Duration", min_width=9, justify="right")
    table.add_column("Details", min_width=30)

    for i, (stage, st) in enumerate(zip(_STAGES, stage_states), 1):
        status_str = st["status"]
        icon = stage["icon"]
        label = stage["label"]

        if status_str == "waiting":
            status_cell = _STATUS_WAITING
            dur_cell = Text("—", style="dim")
            detail_cell = Text(stage["desc"], style="dim italic")

        elif status_str == "running":
            status_cell = Text(f"{spinner_frame}  Running…", style="bold yellow")
            elapsed = time.perf_counter() - st.get("started_at", time.perf_counter())
            dur_cell = Text(f"{elapsed:.1f}s", style="yellow")
            detail_cell = Text(stage["desc"], style="yellow italic")

        elif status_str == "done":
            status_cell = _STATUS_DONE_OK
            dur_cell = Text(f"{st.get('duration', 0):.1f}s", style="green")
            detail_cell = Text(st.get("detail_str", ""), style="dim")

        else:  # error
            status_cell = _STATUS_DONE_ERR
            dur_cell = Text(f"{st.get('duration', 0):.1f}s", style="red")
            detail_cell = Text(st.get("detail_str", "error"), style="red dim")

        table.add_row(
            f"{i}",
            f"{icon}  {label}",
            status_cell,
            dur_cell,
            detail_cell,
        )

    return table


def _node_detail_str(node: str, partial_state: dict[str, Any]) -> str:
    """Produce a one-line detail string for a completed node."""
    if node == "research":
        sources = len(partial_state.get("research_sources", []))
        findings = len(partial_state.get("research_findings", []))
        backend = partial_state.get("research_backend", "?")
        ok = partial_state.get("research_success", False)
        return (
            f"{sources} source(s), {findings} finding(s) via {backend}"
            if ok
            else f"failed — {partial_state.get('error', '')[:60]}"
        )
    if node == "summarize":
        chars = len(partial_state.get("summary", ""))
        return f"{chars:,} chars" if chars else "empty output"
    if node == "write_report":
        chars = len(partial_state.get("report", ""))
        rev = partial_state.get("revision_count", 0)
        label = f"revision {rev}" if rev else "first draft"
        return f"{chars:,} chars ({label})"
    if node == "quality_check":
        score = partial_state.get("quality_score", 0)
        verdict = partial_state.get("quality_verdict", "?")
        revisions = partial_state.get("revision_count", 0)
        return f"score {score}/100 — {verdict} (rev #{revisions})"
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Spinner frames helper (no external dependency)
# ─────────────────────────────────────────────────────────────────────────────

_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def _spinner_frame(t: float) -> str:
    idx = int(t * 8) % len(_SPINNER_FRAMES)
    return _SPINNER_FRAMES[idx]


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph runner with live progress
# ─────────────────────────────────────────────────────────────────────────────


def run_langgraph_with_progress(
    topic: str,
    output_dir: str,
    model: str,
    quality_threshold: int,
    max_revisions: int,
    thread_id: str,
) -> tuple[dict[str, Any], str | None]:
    """Stream the LangGraph pipeline, showing live progress via Rich Live.

    Returns:
        ``(final_state, report_path)`` where *report_path* is ``None`` on
        failure.
    """
    from src.workflows.langgraph_workflow import ResearchGraph

    graph = ResearchGraph(
        model=model,
        quality_threshold=quality_threshold,
        max_revisions=max_revisions,
        db_path="workflow_state.db",
        verbose=False,
    )

    # Initialise per-stage state: all waiting
    stage_states: list[dict[str, Any]] = [
        {"status": "waiting"} for _ in _STAGES
    ]

    console.print()
    console.print(
        Panel(
            f"[bold cyan]Topic:[/bold cyan] {topic}",
            title="[bold]Multi-Agent Research Pipeline[/bold]",
            subtitle=f"[dim]thread {thread_id[:8]}…[/dim]",
            border_style="cyan",
            padding=(0, 2),
        )
    )
    console.print()

    final_state: dict[str, Any] = {}
    completed_nodes: list[str] = []
    pipeline_start = time.perf_counter()

    # Expected node order drives "which stage is next"
    expected_order = [s["node"] for s in _STAGES]

    with Live(
        _build_progress_table(stage_states),
        console=console,
        refresh_per_second=12,
        transient=False,
    ) as live:
        # Mark first stage as running
        stage_states[0]["status"] = "running"
        stage_states[0]["started_at"] = time.perf_counter()
        live.update(_build_progress_table(stage_states, _spinner_frame(0)))

        try:
            for node_name, partial_state in graph.stream(topic, thread_id=thread_id):
                now = time.perf_counter()

                idx = _NODE_TO_IDX.get(node_name)
                if idx is None:
                    live.update(_build_progress_table(stage_states, _spinner_frame(now)))
                    continue

                # Mark this stage done
                started = stage_states[idx].get("started_at", now)
                stage_states[idx]["status"] = "done"
                stage_states[idx]["duration"] = now - started
                stage_states[idx]["detail_str"] = _node_detail_str(
                    node_name, partial_state
                )
                completed_nodes.append(node_name)
                final_state.update(partial_state)

                logger.info(
                    "[Progress] %s done in %.1fs — %s",
                    node_name,
                    stage_states[idx]["duration"],
                    stage_states[idx]["detail_str"],
                )

                # Activate the next expected stage
                next_expected_idx = len(completed_nodes)
                if next_expected_idx < len(expected_order):
                    next_node = expected_order[next_expected_idx]
                    next_idx = _NODE_TO_IDX[next_node]
                    stage_states[next_idx]["status"] = "running"
                    stage_states[next_idx]["started_at"] = now
                elif node_name == "quality_check":
                    verdict = partial_state.get("quality_verdict", "")
                    score = partial_state.get("quality_score", 0)
                    revisions = partial_state.get("revision_count", 0)
                    # Revision loop: re-activate write_report
                    if verdict == "REVISION_REQUIRED" and revisions < max_revisions:
                        wr_idx = _NODE_TO_IDX["write_report"]
                        stage_states[wr_idx]["status"] = "running"
                        stage_states[wr_idx]["started_at"] = now
                        stage_states[wr_idx]["detail_str"] = (
                            f"revision {revisions}/{max_revisions} "
                            f"(prev score: {score}/100)"
                        )

                live.update(
                    _build_progress_table(stage_states, _spinner_frame(now))
                )

        except KeyboardInterrupt:
            # Mark any running stages as errored
            for st in stage_states:
                if st["status"] == "running":
                    st["status"] = "error"
                    st["detail_str"] = "interrupted"
            live.update(_build_progress_table(stage_states))
            raise

        except Exception as exc:
            # Mark running stage as error
            for st in stage_states:
                if st["status"] == "running":
                    st["status"] = "error"
                    st["detail_str"] = str(exc)[:80]
            live.update(_build_progress_table(stage_states))
            logger.error("Pipeline error: %s", exc, exc_info=True)
            raise

        # Ensure all stages show "done" (in case a stage was skipped somehow)
        for st in stage_states:
            if st["status"] == "running":
                st["status"] = "done"
                st["duration"] = time.perf_counter() - st.get("started_at", 0)
        live.update(_build_progress_table(stage_states))

    # ── Save report ───────────────────────────────────────────────────────────
    pipeline_duration = time.perf_counter() - pipeline_start
    report_path: str | None = None
    report = final_state.get("report", "")

    if report:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = "".join(c if c.isalnum() or c in "-_ " else "" for c in topic[:40])
        slug = slug.strip().replace(" ", "_")
        filename = f"report_{slug}_{timestamp}.md"
        dest = out_dir / filename
        dest.write_text(report, encoding="utf-8")
        report_path = str(dest)
        logger.info("Report saved → %s", dest)

    final_state["_pipeline_duration"] = pipeline_duration
    return final_state, report_path


# ─────────────────────────────────────────────────────────────────────────────
# CrewAI runner (non-streaming, simple spinner)
# ─────────────────────────────────────────────────────────────────────────────


def run_crewai_with_progress(topic: str, output_dir: str) -> tuple[dict[str, Any], str | None]:
    """Run the CrewAI pipeline with a simple spinner (no node-level streaming)."""
    from src.workflows.research_workflow import ResearchWorkflow

    console.print()
    console.print(
        Panel(
            f"[bold cyan]Topic:[/bold cyan] {topic}",
            title="[bold]CrewAI Research Pipeline[/bold]",
            border_style="cyan",
            padding=(0, 2),
        )
    )
    console.print()

    start = time.perf_counter()
    result_dict: dict[str, Any] = {}
    report_path: str | None = None

    with console.status(
        "[bold yellow]Running CrewAI pipeline (this may take a few minutes)…",
        spinner="dots",
    ):
        workflow = ResearchWorkflow()
        result = workflow.run(topic)

    duration = time.perf_counter() - start

    if result.report:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = "".join(c if c.isalnum() or c in "-_ " else "" for c in topic[:40])
        slug = slug.strip().replace(" ", "_")
        dest = out_dir / f"crewai_report_{slug}_{timestamp}.md"
        dest.write_text(result.report, encoding="utf-8")
        report_path = str(dest)

    result_dict = {
        "topic": topic,
        "report": result.report,
        "summary": result.summary,
        "quality_review": result.quality_review,
        "research_success": result.success,
        "_pipeline_duration": duration,
    }
    return result_dict, report_path


# ─────────────────────────────────────────────────────────────────────────────
# Stats display
# ─────────────────────────────────────────────────────────────────────────────


def print_stats(
    state: dict[str, Any],
    report_path: str | None,
    model: str,
    backend: str,
) -> None:
    """Print execution statistics and token estimates."""
    console.print()
    console.print(Rule("[bold]Run Statistics[/bold]", style="cyan"))
    console.print()

    duration = state.get("_pipeline_duration", 0.0)
    quality_score = state.get("quality_score", 0)
    quality_verdict = state.get("quality_verdict", "N/A")
    revisions = state.get("revision_count", 0)

    # ── Quality table ─────────────────────────────────────────────────────────
    quality_table = Table(
        show_header=False, box=None, padding=(0, 2), expand=False
    )
    quality_table.add_column("Key", style="dim", min_width=20)
    quality_table.add_column("Value", style="bold")

    quality_table.add_row("Backend", backend.upper())
    quality_table.add_row("Total duration", f"{duration:.1f}s")
    quality_table.add_row("Revisions", str(revisions))

    verdict_style = "bold green" if quality_verdict == "APPROVED" else "bold yellow"
    quality_table.add_row(
        "Quality verdict",
        Text(quality_verdict, style=verdict_style),
    )
    if quality_score:
        bar_filled = int(quality_score / 5)  # 0-20 blocks for 0-100 score
        bar = "█" * bar_filled + "░" * (20 - bar_filled)
        score_style = (
            "bold green" if quality_score >= 70 else
            "bold yellow" if quality_score >= 50 else
            "bold red"
        )
        quality_table.add_row(
            "Quality score",
            Text(f"{quality_score:3d}/100  {bar}", style=score_style),
        )

    # ── Token estimates ───────────────────────────────────────────────────────
    token_table = Table(
        title="[bold]Estimated Token Usage[/bold] [dim](output only)[/dim]",
        show_header=True,
        header_style="bold cyan",
        box=None,
        padding=(0, 2),
        expand=False,
    )
    token_table.add_column("Stage", min_width=24)
    token_table.add_column("Output tokens", justify="right", min_width=14)
    token_table.add_column("~Input tokens", justify="right", min_width=14, style="dim")

    stage_tokens = _estimate_stage_tokens(state, model)
    total_out = 0
    total_in = 0
    for label, toks in stage_tokens.items():
        out = toks["output"]
        inp = toks["input"]
        total_out += out
        total_in += inp
        token_table.add_row(label, f"{out:,}", f"~{inp:,}")

    token_table.add_row(
        "[bold]Total[/bold]",
        f"[bold]{total_out:,}[/bold]",
        f"[bold dim]~{total_in:,}[/bold dim]",
    )

    console.print(Columns([quality_table, token_table], equal=False, expand=False))

    # ── Quality dimensions ────────────────────────────────────────────────────
    dims = state.get("quality_dimensions", [])
    if dims:
        console.print()
        dim_table = Table(
            title="[bold]Quality Breakdown[/bold]",
            show_header=True,
            header_style="bold cyan",
            box=None,
            padding=(0, 2),
        )
        dim_table.add_column("Dimension", min_width=28)
        dim_table.add_column("Score", justify="center", min_width=8)
        dim_table.add_column("Bar", min_width=22)
        dim_table.add_column("Status", min_width=8)

        for d in dims:
            score = d.get("score", 0)
            max_s = d.get("max_score", 25)
            pct = int((score / max_s) * 100) if max_s else 0
            bar_len = int(pct / 5)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            passed = pct >= 70
            status = Text("✓ Pass", style="green") if passed else Text("✗ Fail", style="red")
            bar_style = "green" if passed else ("yellow" if pct >= 40 else "red")
            dim_table.add_row(
                d.get("display_name", d.get("name", "?")),
                f"{score}/{max_s}",
                Text(bar, style=bar_style),
                status,
            )
        console.print(dim_table)

    # ── Output path ───────────────────────────────────────────────────────────
    console.print()
    if report_path:
        console.print(
            Panel(
                f"[bold green]✓[/bold green] Report saved to [cyan]{report_path}[/cyan]",
                border_style="green",
                padding=(0, 2),
            )
        )
    else:
        console.print(
            Panel(
                "[bold red]✗[/bold red] No report was generated.",
                border_style="red",
                padding=(0, 2),
            )
        )
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# Report display
# ─────────────────────────────────────────────────────────────────────────────


def print_report(report: str, topic: str) -> None:
    """Render the final Markdown report to the console."""
    if not report:
        console.print("[dim italic]No report content to display.[/dim italic]")
        return

    console.print()
    console.print(Rule("[bold]Final Report[/bold]", style="cyan"))
    console.print()
    # Limit console display to first 6 000 chars to avoid flooding terminals;
    # the full report is always saved to disk.
    display = report if len(report) <= 6_000 else report[:6_000] + "\n\n…[truncated — see saved file]"
    try:
        console.print(Markdown(display))
    except Exception:
        # Fallback to plain text if Markdown rendering fails
        console.print(display)
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# Error helpers
# ─────────────────────────────────────────────────────────────────────────────


def _print_error(title: str, message: str, hint: str = "") -> None:
    """Print a formatted error panel to stderr."""
    body = f"[bold red]{message}[/bold red]"
    if hint:
        body += f"\n\n[dim]{hint}[/dim]"
    err_console.print(
        Panel(body, title=f"[bold red]{title}[/bold red]", border_style="red", padding=(0, 2))
    )


_KNOWN_ERRORS: list[tuple[type[Exception], str, str]] = [
    (
        KeyboardInterrupt,
        "Interrupted",
        "Run was cancelled by the user (Ctrl-C).",
        "The workflow state has been saved. Re-run with the same --thread-id to resume.",
    ),
    (
        ImportError,
        "Missing dependency",
        "",
        "Run: pip install -r requirements.txt",
    ),
]


def _friendly_error(exc: BaseException) -> tuple[str, str, str]:
    """Return (title, message, hint) for a known or unknown exception."""
    exc_str = str(exc)

    if isinstance(exc, KeyboardInterrupt):
        return ("Interrupted", "Run cancelled by user (Ctrl-C).",
                "Workflow state saved — resume with the same --thread-id.")

    if isinstance(exc, FileNotFoundError):
        return ("File not found", exc_str, "Check that your output directory is writable.")

    if "OPENAI_API_KEY" in exc_str or "AuthenticationError" in type(exc).__name__:
        return (
            "OpenAI API key error",
            exc_str,
            "Set OPENAI_API_KEY in your .env file.  "
            "Copy .env.example → .env and fill in your key.",
        )
    if "RateLimitError" in type(exc).__name__ or "rate limit" in exc_str.lower():
        return (
            "OpenAI rate limit hit",
            exc_str,
            "Wait a moment and retry.  "
            "Consider upgrading your OpenAI plan for higher throughput.",
        )
    if "quota" in exc_str.lower() or "insufficient_quota" in exc_str:
        return (
            "OpenAI quota exceeded",
            exc_str,
            "Your OpenAI account has run out of credits.  "
            "Add billing at platform.openai.com.",
        )
    if "ConnectionError" in type(exc).__name__ or "timeout" in exc_str.lower():
        return (
            "Network error",
            exc_str,
            "Check your internet connection and retry.",
        )
    if "serper" in exc_str.lower() or "SERPER_API_KEY" in exc_str:
        return (
            "Serper search error",
            exc_str,
            "Check your SERPER_API_KEY in .env.  "
            "The system will fall back to DuckDuckGo if no key is set.",
        )
    return ("Unexpected error", exc_str, "Check logs/ for the full traceback.")


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="research-system",
        description="Multi-Agent Autonomous Research System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "topic",
        help="Research topic or question to investigate",
    )
    parser.add_argument(
        "--backend",
        choices=["langgraph", "crewai"],
        default="langgraph",
        help="Orchestration backend (default: langgraph)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="OpenAI model to use for all agents (default: gpt-4o)",
    )
    parser.add_argument(
        "--quality-threshold",
        type=int,
        default=70,
        metavar="N",
        help="Minimum quality score 0–100 to approve a report (default: 70)",
    )
    parser.add_argument(
        "--max-revisions",
        type=int,
        default=3,
        metavar="N",
        help="Maximum revision cycles before forcing output (default: 3)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory to save the final report (default: outputs/)",
    )
    parser.add_argument(
        "--thread-id",
        default=None,
        help="Resume a previous LangGraph run using this checkpoint ID",
    )
    parser.add_argument(
        "--save-metadata",
        action="store_true",
        help="Save a JSON metadata file alongside the report",
    )
    parser.add_argument(
        "--no-report-preview",
        action="store_true",
        help="Skip printing the report to the console",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="WARNING",
        help="Logging verbosity (default: WARNING — keeps console clean)",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    """Application entry point.  Returns exit code (0 = success, 1 = error)."""
    args = parse_args()

    # ── Logging ───────────────────────────────────────────────────────────────
    from src.config import setup_logging

    setup_logging(log_level=args.log_level, log_dir="logs")

    # Silence noisy third-party loggers so the Rich display stays clean.
    # This is done regardless of the user's --log-level so the console is
    # never polluted by LiteLLM/httpx debug chatter.
    _SILENT_LOGGERS = (
        "src", "crewai", "langchain", "langgraph",
        "litellm", "LiteLLM", "openai", "httpx", "httpcore",
        "urllib3", "requests", "aiohttp", "anthropic",
    )
    _console_level = getattr(logging, args.log_level.upper(), logging.WARNING)
    for _name in _SILENT_LOGGERS:
        _lg = logging.getLogger(_name)
        # Only silence for console handler; file handler keeps DEBUG
        _lg.setLevel(max(_console_level, logging.WARNING))

    # LiteLLM sometimes reconfigures the root logger — pin it here too.
    try:
        import litellm  # noqa: PLC0415

        litellm.suppress_debug_info = True
        litellm.set_verbose = False
        logging.getLogger("litellm").setLevel(logging.ERROR)
        logging.getLogger("LiteLLM").setLevel(logging.ERROR)
    except Exception:
        pass

    # ── Header ────────────────────────────────────────────────────────────────
    console.print()
    console.print(
        Rule(
            "[bold cyan]Multi-Agent Autonomous Research System[/bold cyan]",
            style="cyan",
        )
    )

    thread_id = args.thread_id or str(uuid.uuid4())
    output_dir = args.output_dir

    # ── Run pipeline ──────────────────────────────────────────────────────────
    final_state: dict[str, Any] = {}
    report_path: str | None = None

    try:
        if args.backend == "langgraph":
            final_state, report_path = run_langgraph_with_progress(
                topic=args.topic,
                output_dir=output_dir,
                model=args.model,
                quality_threshold=args.quality_threshold,
                max_revisions=args.max_revisions,
                thread_id=thread_id,
            )
        else:
            final_state, report_path = run_crewai_with_progress(
                topic=args.topic,
                output_dir=output_dir,
            )

    except KeyboardInterrupt:
        title, msg, hint = _friendly_error(KeyboardInterrupt())
        _print_error(title, msg, hint)
        console.print(f"\n[dim]Thread ID for resumption: [cyan]{thread_id}[/cyan][/dim]\n")
        return 130

    except Exception as exc:  # noqa: BLE001
        title, msg, hint = _friendly_error(exc)
        _print_error(title, msg, hint)
        logger.error("Fatal error", exc_info=True)
        return 1

    # ── Print report ──────────────────────────────────────────────────────────
    report = final_state.get("report", "")
    if not args.no_report_preview:
        print_report(report, args.topic)

    # ── Stats ─────────────────────────────────────────────────────────────────
    print_stats(
        state=final_state,
        report_path=report_path,
        model=args.model,
        backend=args.backend,
    )

    # ── Metadata ──────────────────────────────────────────────────────────────
    if args.save_metadata:
        meta: dict[str, Any] = {
            "topic": args.topic,
            "backend": args.backend,
            "model": args.model,
            "thread_id": thread_id,
            "timestamp": datetime.now().isoformat(),
            "quality_score": final_state.get("quality_score", 0),
            "quality_verdict": final_state.get("quality_verdict", ""),
            "revision_count": final_state.get("revision_count", 0),
            "pipeline_duration_secs": round(final_state.get("_pipeline_duration", 0), 2),
            "report_path": report_path,
            "research_backend": final_state.get("research_backend", ""),
            "sources_count": len(final_state.get("research_sources", [])),
            "findings_count": len(final_state.get("research_findings", [])),
        }
        meta_dir = Path(output_dir)
        meta_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        meta_path = meta_dir / f"metadata_{ts}.json"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        console.print(f"[dim]Metadata → {meta_path}[/dim]")

    # ── Thread ID hint (for resumption) ───────────────────────────────────────
    if args.backend == "langgraph":
        console.print(
            f"[dim]Resume this run:  "
            f"python main.py \"...\" --thread-id {thread_id}[/dim]\n"
        )

    success = bool(report)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
