"""
Search & Reader Agent — finds and scrapes live web data about a competitor.

Uses Tavily search to discover URLs, then scrapes the most relevant pages
with BeautifulSoup to extract clean text for analysis.
"""

from langchain_core.messages import SystemMessage, HumanMessage
from src.config import get_llm
from src.state import AgentState
from src.tools.web_search import search_web
from src.tools.web_scraper import scrape_url

SEARCH_AGENT_PROMPT = """You are a Search & Reader agent specializing in competitive intelligence.

Given a competitor name and focus areas, generate 2-3 targeted search queries that will
find the most relevant, recent information. Think about:
- Recent product launches or updates
- Pricing changes
- Technical blog posts or engineering announcements
- Funding, partnerships, or acquisitions
- User reviews and complaints

Respond with a JSON array of search query strings. Example:
["Notion AI features 2025", "Notion pricing changes recent", "Notion vs competitors review"]

ONLY output the JSON array, nothing else."""


SUMMARIZE_PROMPT = """You are a research analyst. Summarize the following scraped web content
into key intelligence points about {competitor}. Focus on:
- Product features and recent changes
- Pricing and packaging
- Technical architecture mentions
- Market positioning and messaging
- Any weaknesses or user complaints

Be concise and factual. Cite the source URL for each point.

Content to analyze:
{content}"""


def search_reader_node(state: AgentState) -> dict:
    """Execute web search and scraping for competitor intelligence."""
    llm = get_llm(temperature=0.3, max_tokens=500)
    competitor = state.get("competitor_name", "Unknown")
    focus_areas = state.get("focus_areas", ["general"])

    errors: list[str] = []
    all_search_results: list[dict] = []
    all_scraped: list[dict] = []

    # ── Step 1: Generate search queries via LLM ─────────────────────────
    try:
        query_response = llm.invoke([
            SystemMessage(content=SEARCH_AGENT_PROMPT),
            HumanMessage(content=f"Competitor: {competitor}\nFocus areas: {', '.join(focus_areas)}"),
        ])
        import json
        raw = query_response.content.strip()
        # Handle markdown code blocks
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        queries = json.loads(raw)
    except Exception as e:
        # Fallback queries if LLM fails
        queries = [
            f"{competitor} product features 2025",
            f"{competitor} pricing plans",
            f"{competitor} recent news",
        ]
        errors.append(f"Query generation fallback: {e}")

    # ── Step 2: Execute searches ────────────────────────────────────────
    for query in queries[:3]:  # Cap at 3 to control costs
        try:
            results = search_web.invoke({"query": query})
            if isinstance(results, list):
                all_search_results.extend(results)
        except Exception as e:
            errors.append(f"Search failed for '{query}': {e}")

    # ── Step 3: Scrape top URLs ─────────────────────────────────────────
    seen_urls: set[str] = set()
    urls_to_scrape: list[str] = []

    for result in all_search_results:
        url = result.get("url", "")
        if url and url not in seen_urls and not result.get("error"):
            seen_urls.add(url)
            urls_to_scrape.append(url)

    for url in urls_to_scrape[:4]:  # Scrape max 4 pages
        try:
            scraped = scrape_url.invoke({"url": url})
            if scraped.get("success"):
                all_scraped.append(scraped)
        except Exception as e:
            errors.append(f"Scrape failed for {url}: {e}")

    return {
        "search_results": all_search_results,
        "scraped_content": all_scraped,
        "completed_agents": ["search_reader"],
        "errors": errors,
    }
