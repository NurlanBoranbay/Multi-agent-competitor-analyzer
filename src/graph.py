"""
LangGraph workflow — wires all agents into a conditional state graph.

Flow:
  START → supervisor → (search_reader | visual_auditor | writer) → ...
  writer → critic → (writer revision | FINISH)
"""

from langgraph.graph import StateGraph, END
from src.state import AgentState
from src.agents.supervisor import supervisor_node
from src.agents.search_reader import search_reader_node
from src.agents.visual_auditor import visual_auditor_node
from src.agents.writer import writer_node
from src.agents.critic import critic_node


def _route_supervisor(state: AgentState) -> str:
    """Route from supervisor to the correct worker node."""
    next_agent = state.get("next_agent", "FINISH")
    if next_agent == "FINISH":
        return "end"
    return next_agent


def _route_critic(state: AgentState) -> str:
    """Route from critic — either back to writer for revision or to end."""
    if state.get("critic_pass"):
        return "end"
    return "writer"


def build_graph() -> StateGraph:
    """Construct and compile the multi-agent LangGraph."""

    graph = StateGraph(AgentState)

    # ── Add nodes ───────────────────────────────────────────────────────
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("search_reader", search_reader_node)
    graph.add_node("visual_auditor", visual_auditor_node)
    graph.add_node("writer", writer_node)
    graph.add_node("critic", critic_node)

    # ── Set entry point ─────────────────────────────────────────────────
    graph.set_entry_point("supervisor")

    # ── Conditional routing from supervisor ──────────────────────────────
    graph.add_conditional_edges(
        "supervisor",
        _route_supervisor,
        {
            "search_reader": "search_reader",
            "visual_auditor": "visual_auditor",
            "writer": "writer",
            "end": END,
        },
    )

    # ── Worker nodes always return to supervisor ────────────────────────
    graph.add_edge("search_reader", "supervisor")
    graph.add_edge("visual_auditor", "supervisor")

    # ── Writer → Critic → (Writer | END) loop ──────────────────────────
    graph.add_edge("writer", "critic")
    graph.add_conditional_edges(
        "critic",
        _route_critic,
        {
            "writer": "writer",
            "end": END,
        },
    )

    return graph.compile()
