"""
Critic Agent — reviews the draft report against strict quality criteria.

If the report fails any criteria, it provides specific feedback for the
Writer to revise. Implements the quality gate in the Writer-Critic loop.
"""

from langchain_core.messages import SystemMessage, HumanMessage
from src.config import get_llm, MAX_CRITIC_RETRIES
from src.state import AgentState


CRITIC_PROMPT = """You are a strict quality reviewer for competitive intelligence reports.

Review the report below against these criteria:

1. **Source Citations**: Are ALL factual claims backed by a source URL? (Critical)
2. **SWOT Completeness**: Does it have at least 3 bullet points per SWOT category?
3. **Structure**: Does it follow the required section format (Executive Summary, Company Overview, SWOT, Product Analysis, Pricing, Visual Assessment, Recommendations, Sources)?
4. **Actionability**: Are the recommendations specific and actionable (not generic)?
5. **Tone**: Is the tone professional and analytical (not promotional)?
6. **Data Gaps**: Are missing data points acknowledged rather than fabricated?

## Response Format
Respond with a JSON object:
{{
    "pass": true/false,
    "score": 0-100,
    "issues": ["issue 1 description", "issue 2 description"],
    "summary": "Overall assessment in 1-2 sentences"
}}

If score >= 70, set "pass" to true. Otherwise false.
ONLY output the JSON, nothing else."""


def critic_node(state: AgentState) -> dict:
    """Review the draft report and decide if it passes quality criteria."""
    llm = get_llm(temperature=0.0, max_tokens=500)
    draft = state.get("draft_report", "")
    revision_count = state.get("revision_count", 0)

    # If we've hit max retries, just accept the report
    if revision_count >= MAX_CRITIC_RETRIES:
        return {
            "critic_pass": True,
            "critic_feedback": "Accepted after maximum revision attempts.",
            "final_report": draft,
        }

    if not draft:
        return {
            "critic_pass": False,
            "critic_feedback": "No draft report was generated.",
            "errors": ["Critic received empty draft"],
        }

    try:
        response = llm.invoke([
            SystemMessage(content=CRITIC_PROMPT),
            HumanMessage(content=f"## Report to Review:\n\n{draft}"),
        ])

        import json
        raw = response.content.strip()
        # Handle markdown code blocks
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        review = json.loads(raw)

        passed = review.get("pass", False)
        issues = review.get("issues", [])
        score = review.get("score", 0)
        summary = review.get("summary", "")

        feedback = f"Score: {score}/100. {summary}"
        if issues:
            feedback += "\n\nIssues to fix:\n" + "\n".join(f"- {i}" for i in issues)

        return {
            "critic_pass": passed,
            "critic_feedback": feedback,
            "final_report": draft if passed else "",
        }

    except Exception as e:
        # If critic fails to parse, accept the draft
        return {
            "critic_pass": True,
            "critic_feedback": f"Critic parsing error ({e}), accepting draft.",
            "final_report": draft,
        }
