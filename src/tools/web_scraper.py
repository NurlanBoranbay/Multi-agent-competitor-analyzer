"""
Web scraping tool using httpx + BeautifulSoup.

Fetches a URL, strips boilerplate, and returns clean text for analysis.
Falls back gracefully when pages block bots or return malformed HTML.
"""

import httpx
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from src.config import SCRAPE_TIMEOUT_SEC


_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}

# Tags whose text we actually want
_KEEP_TAGS = {"p", "h1", "h2", "h3", "h4", "li", "td", "th", "blockquote", "span", "a"}

# Tags to remove entirely (including children)
_STRIP_TAGS = {"script", "style", "nav", "footer", "header", "aside", "form", "noscript", "svg"}


def _clean_html(html: str, max_chars: int = 8000) -> str:
    """Parse HTML and return the most relevant text content."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove noise elements
    for tag in soup.find_all(_STRIP_TAGS):
        tag.decompose()

    # Extract text from useful tags
    chunks: list[str] = []
    for tag in soup.find_all(_KEEP_TAGS):
        text = tag.get_text(separator=" ", strip=True)
        if len(text) > 20:  # skip tiny fragments
            chunks.append(text)

    combined = "\n".join(chunks)
    return combined[:max_chars]


@tool
def scrape_url(url: str) -> dict:
    """Fetch and extract clean text content from a web page.

    Args:
        url: The full URL to scrape.

    Returns:
        A dict with keys: url, title, content, success.
    """
    try:
        with httpx.Client(
            headers=_HEADERS,
            timeout=SCRAPE_TIMEOUT_SEC,
            follow_redirects=True,
        ) as client:
            resp = client.get(url)
            resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        title = soup.title.string.strip() if soup.title and soup.title.string else url
        content = _clean_html(resp.text)

        return {
            "url": url,
            "title": title,
            "content": content,
            "success": True,
        }
    except Exception as e:
        return {
            "url": url,
            "title": "",
            "content": "",
            "success": False,
            "error": str(e),
        }
