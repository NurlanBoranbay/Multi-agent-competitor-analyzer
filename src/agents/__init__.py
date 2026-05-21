"""
agents — Specialized worker nodes for the intelligence network.

Each module exports a single node function compatible with LangGraph:
    supervisor_node, search_reader_node, visual_auditor_node,
    writer_node, critic_node
"""

from src.agents.supervisor import supervisor_node
from src.agents.search_reader import search_reader_node
from src.agents.visual_auditor import visual_auditor_node
from src.agents.writer import writer_node
from src.agents.critic import critic_node

__all__ = [
    "supervisor_node",
    "search_reader_node",
    "visual_auditor_node",
    "writer_node",
    "critic_node",
]
