"""
Tavily-powered web search tool.

Wraps TavilySearchResults so the Search & Reader agent can query the web
for recent competitor intelligence.
"""

from langchain_core.tools import tool
from tavily import TavilyClient
from src.config import TAVILY_API_KEY, SEARCH_MAX_RESULTS


@tool
def search_web(query: str) -> list[dict]:
    """Search the web for recent information about a topic.

    Args:
        query: The search query string.

    Returns:
        A list of dicts with keys: title, url, content, score.
    """
    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)
        response = client.search(
            query=query,
            max_results=SEARCH_MAX_RESULTS,
            search_depth="basic",           # "basic" is cheaper than "advanced"
            include_raw_content=False,
        )
        results = []
        for r in response.get("results", []):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
                "score": r.get("score", 0.0),
            })
        return results
    except Exception as e:
        return [{"error": str(e)}]
