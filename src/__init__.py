"""
src — Multi-Agent Competitor Intelligence Network.

Orchestrates a team of specialized AI agents (search, visual audit,
writer, critic) via a LangGraph supervisor to compile SWOT reports.
"""

from src.graph import build_graph
from src.state import AgentState

__all__ = ["build_graph", "AgentState"]
