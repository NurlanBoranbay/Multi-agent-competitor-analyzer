"""
Shared state schema for the multi-agent graph.

Every node reads from / writes to this TypedDict so agents can coordinate
through a single, transparent data structure.
"""

from __future__ import annotations
from typing import Annotated, TypedDict
import operator


def _merge_lists(left: list, right: list) -> list:
    """Reducer that appends new items to an existing list."""
    return left + right


class AgentState(TypedDict):
    """Central state passed between all nodes in the LangGraph."""

    # ── User input ──────────────────────────────────────────────────────
    competitor_name: str                # e.g. "Notion"
    focus_areas: list[str]              # e.g. ["pricing", "AI features"]

    # ── Supervisor routing ──────────────────────────────────────────────
    next_agent: str                     # which node to call next
    completed_agents: Annotated[list[str], _merge_lists]

    # ── Search & Reader outputs ─────────────────────────────────────────
    search_results: Annotated[list[dict], _merge_lists]
    scraped_content: Annotated[list[dict], _merge_lists]

    # ── Visual Auditor outputs ──────────────────────────────────────────
    screenshot_paths: Annotated[list[str], _merge_lists]
    visual_analysis: str

    # ── Writer / Critic outputs ─────────────────────────────────────────
    draft_report: str
    critic_feedback: str
    critic_pass: bool
    revision_count: int

    # ── Final output ────────────────────────────────────────────────────
    final_report: str
    errors: Annotated[list[str], _merge_lists]
