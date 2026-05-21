"""
Writer Agent — compiles all gathered intelligence into a structured SWOT report.

Takes search results, scraped content, and visual analysis and produces a
comprehensive markdown report with proper citations.
"""

from langchain_core.messages import SystemMessage, HumanMessage
from src.config import get_llm
from src.state import AgentState


WRITER_PROMPT = """You are an expert competitive intelligence analyst and business writer.

Using the research data provided, write a comprehensive **Competitor Intelligence Report**
in markdown format. The report MUST follow this exact structure:

# Competitor Intelligence Report: {competitor}
*Generated: {date}*

## Executive Summary
(2-3 paragraph high-level overview)

## Company Overview
(What they do, target market, founding year if found, key stats)

## SWOT Analysis

### Strengths
- (bullet points with source citations)

### Weaknesses
- (bullet points with source citations)

### Opportunities
- (market gaps, trends they could exploit)

### Threats
- (competitive pressures, market risks)

## Product & Feature Analysis
(Key features, recent updates, technical differentiators)

## Pricing & Packaging
(Tiers, pricing strategy, free vs paid analysis)

## Visual & UX Assessment
(Based on visual auditor findings — UI quality, design trends, CTA effectiveness)

## Key Takeaways & Recommendations
(3-5 actionable recommendations for competing against them)

## Sources
(List all URLs referenced)

---

## Rules
- EVERY claim must cite a source URL in brackets like [source](url)
- Be specific with data points — avoid vague language
- If information is unavailable, state "Data not available" rather than guessing
- Keep the tone professional and analytical
- Total length: 1500-2500 words"""


REVISION_PROMPT = """You are revising a competitive intelligence report based on critic feedback.

## Original Report
{draft}

## Critic Feedback
{feedback}

## Instructions
Revise the report to address ALL critic feedback points. Maintain the same structure.
Output the COMPLETE revised report in markdown format."""


def writer_node(state: AgentState) -> dict:
    """Compile all intelligence into a structured SWOT report."""
    llm = get_llm(temperature=0.3, max_tokens=4096)
    competitor = state.get("competitor_name", "Unknown")

    # Check if this is a revision pass
    critic_feedback = state.get("critic_feedback", "")
    existing_draft = state.get("draft_report", "")

    if critic_feedback and existing_draft:
        # ── Revision mode ───────────────────────────────────────────────
        response = llm.invoke([
            SystemMessage(content="You are an expert competitive intelligence analyst."),
            HumanMessage(content=REVISION_PROMPT.format(
                draft=existing_draft,
                feedback=critic_feedback,
            )),
        ])
        return {
            "draft_report": response.content,
            "revision_count": state.get("revision_count", 0) + 1,
        }

    # ── First draft mode ────────────────────────────────────────────────
    # Compile all gathered data into a research brief
    research_sections: list[str] = []

    # Search results
    search_results = state.get("search_results", [])
    if search_results:
        research_sections.append("## Web Search Results")
        for r in search_results:
            if not r.get("error"):
                research_sections.append(
                    f"**{r.get('title', 'No title')}** ({r.get('url', '')})\n"
                    f"{r.get('content', 'No content')}\n"
                )

    # Scraped content
    scraped = state.get("scraped_content", [])
    if scraped:
        research_sections.append("\n## Scraped Page Content")
        for s in scraped:
            research_sections.append(
                f"**{s.get('title', 'No title')}** ({s.get('url', '')})\n"
                f"{s.get('content', '')[:2000]}\n"  # Truncate to save tokens
            )

    # Visual analysis
    visual = state.get("visual_analysis", "")
    if visual:
        research_sections.append(f"\n## Visual/UI Analysis\n{visual}")

    research_brief = "\n\n".join(research_sections) if research_sections else "No research data available."

    from datetime import datetime
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    response = llm.invoke([
        SystemMessage(content=WRITER_PROMPT.format(competitor=competitor, date=date_str)),
        HumanMessage(content=f"Here is all the research data gathered:\n\n{research_brief}"),
    ])

    return {
        "draft_report": response.content,
        "revision_count": 0,
        "completed_agents": ["writer"],
    }
