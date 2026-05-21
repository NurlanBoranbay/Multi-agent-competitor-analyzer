#!/usr/bin/env python3
"""
Multi-Agent Competitor Intelligence Network
============================================
CLI entry point — give it a competitor name and get a SWOT intelligence report.

Usage:
    python main.py "Notion"
    python main.py "Stripe" --focus "pricing,API,developer experience"
    python main.py "Figma" --output reports/figma_report.md
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-Agent Competitor Intelligence Network",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "Notion"
  python main.py "Stripe" --focus "pricing,API,developer experience"
  python main.py "Figma" --output reports/figma_report.md
        """,
    )
    parser.add_argument("competitor", help="Name of the competitor to analyze")
    parser.add_argument(
        "--focus",
        default="product features,pricing,market position,recent updates",
        help="Comma-separated focus areas (default: product features,pricing,market position,recent updates)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path for the report (default: reports/<competitor>_report.md)",
    )
    args = parser.parse_args()

    # ── Validate environment ────────────────────────────────────────────
    console.print(Panel(
        f"[bold cyan] Competitor Intelligence Network[/bold cyan]\n"
        f"[dim]Analyzing: [bold]{args.competitor}[/bold][/dim]",
        border_style="cyan",
    ))

    try:
        from src.config import _check_keys
        _check_keys()
    except EnvironmentError as e:
        console.print(f"[bold red] Configuration Error:[/bold red] {e}")
        sys.exit(1)

    # ── Build and run the graph ─────────────────────────────────────────
    from src.graph import build_graph

    focus_areas = [f.strip() for f in args.focus.split(",")]

    initial_state = {
        "competitor_name": args.competitor,
        "focus_areas": focus_areas,
        "next_agent": "",
        "completed_agents": [],
        "search_results": [],
        "scraped_content": [],
        "screenshot_paths": [],
        "visual_analysis": "",
        "draft_report": "",
        "critic_feedback": "",
        "critic_pass": False,
        "revision_count": 0,
        "final_report": "",
        "errors": [],
    }

    graph = build_graph()

    console.print(f"\n[bold green]▶ Starting agent pipeline...[/bold green]")
    console.print(f"  Focus areas: {', '.join(focus_areas)}\n")

    # Stream events to show progress
    final_state = None
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Initializing agents...", total=None)

        for event in graph.stream(initial_state, {"recursion_limit": 25}):
            # event is a dict like {"node_name": {state_updates}}
            for node_name, updates in event.items():
                if node_name == "supervisor":
                    next_agent = updates.get("next_agent", "")
                    progress.update(task, description=f"Supervisor → routing to [bold]{next_agent}[/bold]")
                elif node_name == "search_reader":
                    n_results = len(updates.get("search_results", []))
                    n_scraped = len(updates.get("scraped_content", []))
                    progress.update(task, description=f"Search & Reader: found {n_results} results, scraped {n_scraped} pages")
                elif node_name == "visual_auditor":
                    n_screenshots = len(updates.get("screenshot_paths", []))
                    progress.update(task, description=f"Visual Auditor: captured {n_screenshots} screenshots")
                elif node_name == "writer":
                    rev = updates.get("revision_count", 0)
                    label = f" (revision {rev})" if rev > 0 else ""
                    progress.update(task, description=f"Writer: composing report{label}...")
                elif node_name == "critic":
                    passed = updates.get("critic_pass", False)
                    icon = "✅" if passed else "🔄"
                    progress.update(task, description=f"Critic: {icon} {'approved' if passed else 'requesting revision'}")

                final_state = {**initial_state, **(final_state or {}), **updates}

        progress.update(task, description="[bold green] Pipeline complete!")

    # ── Output report ───────────────────────────────────────────────────
    report = (final_state or {}).get("final_report", "") or (final_state or {}).get("draft_report", "")

    if not report:
        console.print("[bold red] No report was generated.[/bold red]")
        errors = (final_state or {}).get("errors", [])
        if errors:
            console.print("[yellow]Errors encountered:[/yellow]")
            for e in errors:
                console.print(f"  • {e}")
        sys.exit(1)

    # Save to file
    output_path = args.output or f"reports/{args.competitor.lower().replace(' ', '_')}_report.md"
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(report, encoding="utf-8")

    console.print(f"\n[bold green] Report saved to:[/bold green] {output_path}")

    # Show errors if any
    errors = (final_state or {}).get("errors", [])
    if errors:
        console.print(f"\n[yellow]  {len(errors)} non-fatal errors during execution:[/yellow]")
        for e in errors:
            console.print(f"  [dim]• {e}[/dim]")

    # Print report preview
    console.print("\n" + "─" * 60)
    console.print(Panel(Markdown(report[:3000] + ("\n\n... [truncated]" if len(report) > 3000 else "")),
                        title="Report Preview", border_style="green"))


if __name__ == "__main__":
    main()
