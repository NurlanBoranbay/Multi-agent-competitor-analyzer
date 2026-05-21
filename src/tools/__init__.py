"""
tools — LangChain tool wrappers for web search, scraping, and screenshots.

Each tool is decorated with @tool and can be bound to agents or called directly.
"""

from src.tools.web_search import search_web
from src.tools.web_scraper import scrape_url
from src.tools.screenshot import take_screenshot

__all__ = ["search_web", "scrape_url", "take_screenshot"]
