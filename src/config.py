"""
Configuration module — loads API keys, selects cheap models, and provides
shared constants used across the agent network.
"""

import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()

# ---------------------------------------------------------------------------
# API keys (validated at import time so we fail fast)
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")


def _check_keys() -> None:
    missing = []
    if not ANTHROPIC_API_KEY:
        missing.append("ANTHROPIC_API_KEY")
    if not TAVILY_API_KEY:
        missing.append("TAVILY_API_KEY")
    if missing:
        raise EnvironmentError(
            f"Missing required API keys: {', '.join(missing)}. "
            "Copy .env.example → .env and fill in your keys."
        )


# ---------------------------------------------------------------------------
# Model factory — uses Claude 4.5 Haiku everywhere to keep costs minimal
# Input: $0.25/MTok  Output: $1.25/MTok  (cheapest Claude model)
# ---------------------------------------------------------------------------
MODEL_NAME = "claude-haiku-4-5"
MAX_TOKENS = 4096


def get_llm(temperature: float = 0.2, max_tokens: int = MAX_TOKENS) -> ChatAnthropic:
    """Return a cheap Claude Haiku instance."""
    return ChatAnthropic(
        model=MODEL_NAME,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=ANTHROPIC_API_KEY,
    )


# ---------------------------------------------------------------------------
# Retry / limits
# ---------------------------------------------------------------------------
MAX_CRITIC_RETRIES = 2          # Writer revisions before accepting
SEARCH_MAX_RESULTS = 5          # Tavily results per query
SCRAPE_TIMEOUT_SEC = 15         # Per-page scraping timeout
SCREENSHOT_TIMEOUT_SEC = 20     # Playwright screenshot timeout
