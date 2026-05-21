"""
Supervisor Agent — the orchestrator of the multi-agent network.

Decides which agent to call next based on the current state, and determines
when enough information has been gathered to move to the writing phase.
"""

from langchain_core.messages import SystemMessage, HumanMessage
from src.config import get_llm
from src.state import AgentState

SUPERVISOR_PROMPT = """You are a Supervisor Agent orchestrating a competitive intelligence research team.

Your team consists of:
1. **search_reader** — Searches the web and scrapes pages for competitor data.
2. **visual_auditor** — Takes screenshots of competitor websites and analyses UI/pricing visually.
3. **writer** — Compiles all gathered intelligence into a structured SWOT report.

## Your Job
Given the current state of research, decide which agent should act NEXT.

## Rules
- ALWAYS start with "search_reader" to gather web data first.
- After search_reader completes, send to "visual_auditor" for visual analysis.
- After both search_reader and visual_auditor have completed, send to "writer".
- Once "writer" has completed (and critic has approved), output "FINISH".
- If an agent has already been completed, do NOT send to it again.

## Response Format
Respond with ONLY the next agent name: search_reader, visual_auditor, writer, or FINISH
Nothing else. Just the agent name."""


def supervisor_node(state: AgentState) -> dict:
    """Decide which agent to invoke next."""
    llm = get_llm(temperature=0.0, max_tokens=50)

    completed = state.get("completed_agents", [])
    competitor = state.get("competitor_name", "Unknown")
    focus = ", ".join(state.get("focus_areas", ["general"]))

    status_lines = [
        f"Competitor: {competitor}",
        f"Focus areas: {focus}",
        f"Agents completed so far: {completed if completed else 'none'}",
        f"Search results gathered: {len(state.get('search_results', []))}",
        f"Pages scraped: {len(state.get('scraped_content', []))}",
        f"Screenshots taken: {len(state.get('screenshot_paths', []))}",
        f"Draft report exists: {'yes' if state.get('draft_report') else 'no'}",
        f"Critic approved: {'yes' if state.get('critic_pass') else 'no'}",
    ]

    # Short-circuit logic to save tokens on obvious routing
    if "search_reader" not in completed:
        return {"next_agent": "search_reader"}
    if "visual_auditor" not in completed:
        return {"next_agent": "visual_auditor"}
    if "writer" not in completed:
        return {"next_agent": "writer"}
    if state.get("critic_pass"):
        return {"next_agent": "FINISH"}

    # Fall back to LLM only when routing is ambiguous
    response = llm.invoke([
        SystemMessage(content=SUPERVISOR_PROMPT),
        HumanMessage(content="\n".join(status_lines)),
    ])

    next_agent = response.content.strip().lower()

    # Validate response
    valid = {"search_reader", "visual_auditor", "writer", "FINISH", "finish"}
    if next_agent not in valid:
        next_agent = "FINISH"

    return {"next_agent": next_agent.upper() if next_agent == "finish" else next_agent}
