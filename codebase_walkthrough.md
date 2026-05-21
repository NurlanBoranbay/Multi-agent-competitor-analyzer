# 🕵️ Multi-Agent Competitor Intelligence Network: Codebase Walkthrough

Welcome! This guide is designed to help you understand every single line of code in this multi-agent competitor intelligence network. We will break down how agents operate, how they communicate via shared state, how a supervisor coordinates them using LangGraph, and how to write tools that feed real-world web and visual data into LLMs.

---

## Table of Contents
1. [Core Concepts: Multi-Agent Systems & LangGraph](#1-core-concepts-multi-agent-systems--langgraph)
2. [Shared State Schema (`src/state.py`)](#2-shared-state-schema-srcstatepy)
3. [Configuration and Environment (`src/config.py`)](#3-configuration-and-environment-srcconfigpy)
4. [Tools Layer (`src/tools/`)](#4-tools-layer-srctools)
   - [Web Search (`web_search.py`)](#web-search-web_searchpy)
   - [Web Scraping (`web_scraper.py`)](#web-scraping-web_scraperpy)
   - [Screenshots (`screenshot.py`)](#screenshots-screenshotpy)
5. [The Supervisor Orchestrator (`src/agents/supervisor.py`)](#5-the-supervisor-orchestrator-srcagentssupervisorpy)
6. [Specialized Worker Agents (`src/agents/`)](#6-specialized-worker-agents-srcagents)
   - [Search & Reader (`search_reader.py`)](#search--reader-search_readerpy)
   - [Visual Auditor (`visual_auditor.py`)](#visual-auditor-visual_auditorpy)
   - [Writer (`writer.py`)](#writer-writerpy)
   - [Critic (`critic.py`)](#critic-criticpy)
7. [Graph Assembly & Routing (`src/graph.py`)](#7-graph-assembly--routing-srcgraphpy)
8. [CLI and Execution Flow (`main.py`)](#8-cli-and-execution-flow-mainpy)

---

## 1. Core Concepts: Multi-Agent Systems & LangGraph

In traditional AI applications, a single LLM is given a prompt, runs some tools, and outputs a response. However, for complex tasks like competitor research, a single LLM run often fails because:
* It lacks a structural checklist (it gets distracted or skips steps).
* Large context windows get flooded with raw data, leading to dilution of key insights.
* A single failure (e.g., a scraping failure) can crash the entire chain.

### The Multi-Agent Approach
Instead of one model doing everything, we split the workflow into **specialized workers** coordinated by a **Supervisor Orchestrator**:
1. **Separation of Concerns**: Each agent does one thing exceptionally well (searching, visual analysis, writing, or criticizing).
2. **State Sharing**: Agents do not pass messages directly to each other. Instead, they read from and write to a single, central **State**.
3. **Cyclic Control Loops**: Using LangGraph, we can route the execution backwards (e.g., from Critic back to Writer) to revise work until it meets quality standards.

---

## 2. Shared State Schema (`src/state.py`)

Let's read the code for `src/state.py` line-by-line:

```python
from __future__ import annotations
from typing import Annotated, TypedDict
import operator
```
* **`from __future__ import annotations`**: Enables postponed evaluation of type annotations, letting you use classes defined later in the file as type hints.
* **`Annotated`**: A typing construct that allows attaching metadata or behavior to types. In LangGraph, it is used to attach **reducer functions** to specific state keys.
* **`TypedDict`**: A type-hinted dictionary subclass. It defines the exact keys and value types that our graph state will contain.

```python
def _merge_lists(left: list, right: list) -> list:
    """Reducer that appends new items to an existing list."""
    return left + right
```
* **Reducers**: By default, when a node in LangGraph returns a dictionary containing a state key, it **overwrites** that key in the shared state. A reducer overrides this behavior.
* **`_merge_lists(left, right)`**: This function takes the existing list (`left`) and the new list returned by a node (`right`), adds them together, and returns the combined list. This prevents agents from deleting each other's search results or scraped content!

```python
class AgentState(TypedDict):
    """Central state passed between all nodes in the LangGraph."""

    # ── User input ──────────────────────────────────────────────────────
    competitor_name: str                # e.g. "Notion"
    focus_areas: list[str]              # e.g. ["pricing", "AI features"]

    # ── Supervisor routing ──────────────────────────────────────────────
    next_agent: str                     # which node to call next
    completed_agents: Annotated[list[str], _merge_lists]
```
* **`next_agent`**: Set by the Supervisor to instruct LangGraph where to route the execution flow.
* **`completed_agents`**: An accumulated list of agent names that have completed their execution. We wrap it in `Annotated[..., _merge_lists]` so each completed agent simply appends its name without wiping out previous ones.

```python
    # ── Search & Reader outputs ─────────────────────────────────────────
    search_results: Annotated[list[dict], _merge_lists]
    scraped_content: Annotated[list[dict], _merge_lists]

    # ── Visual Auditor outputs ──────────────────────────────────────────
    screenshot_paths: Annotated[list[str], _merge_lists]
    visual_analysis: str
```
* **`search_results` / `scraped_content`**: Accumulated lists containing dicts from the search engine and beautifulsoup scraper.
* **`screenshot_paths`**: Paths to PNG screenshots taken by Playwright.
* **`visual_analysis`**: Markdown string representing the vision analysis of the screenshots by Claude.

```python
    # ── Writer / Critic outputs ─────────────────────────────────────────
    draft_report: str
    critic_feedback: str
    critic_pass: bool
    revision_count: int

    # ── Final output ────────────────────────────────────────────────────
    final_report: str
    errors: Annotated[list[str], _merge_lists]
```
* **`critic_pass`**: Boolean indicating if the critic approved the draft report.
* **`revision_count`**: Tracks how many times the writer has revised the report to prevent infinite loops.
* **`errors`**: An accumulated list of non-fatal execution errors (e.g., a specific page failed to scrape) so we can display them at the end without halting the entire system.

---

## 3. Configuration and Environment (`src/config.py`)

This file loads environmental keys and provides a central factory for initializing the language model (LLM).

```python
import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()
```
* **`load_dotenv()`**: Reads a local `.env` file and loads its key-value pairs into `os.environ`.

```python
# API keys (validated at import time so we fail fast)
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
```
* **`_check_keys()`**: Ensures the required credentials exist before starting. If not, it raises an error, stopping execution instantly rather than waiting for an API call to crash.

```python
MODEL_NAME = "claude-sonnet-4-20250514"
MAX_TOKENS = 4096

def get_llm(temperature: float = 0.2, max_tokens: int = MAX_TOKENS) -> ChatAnthropic:
    """Return a cheap Claude Haiku instance."""
    return ChatAnthropic(
        model=MODEL_NAME,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=ANTHROPIC_API_KEY,
    )
```
* **`ChatAnthropic`**: The LangChain integration for Anthropic's Claude API.
* **`temperature`**: Determines creativity. We use `0.2` for highly analytical/factual tasks (minimizes hallucinations) and `0.0` for routing decisions.

---

## 4. Tools Layer (`src/tools/`)

Tools are specialized Python functions that the agents call directly to interact with the outside world.

### A. Web Search (`web_search.py`)

This tool uses the **Tavily API**, which is a search engine custom-tailored for LLM applications (returns clean snippets rather than messy HTML).

```python
from langchain_core.tools import tool
from tavily import TavilyClient
from src.config import TAVILY_API_KEY, SEARCH_MAX_RESULTS

@tool
def search_web(query: str) -> list[dict]:
    """Search the web for recent information about a topic..."""
    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)
        response = client.search(
            query=query,
            max_results=SEARCH_MAX_RESULTS,
            search_depth="basic",
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
```
* **`@tool`**: A LangChain decorator that registers the function as a LangChain Tool. This extracts the function's docstring and arguments to create a schema that LLMs can understand.
* **`client.search`**: Hits Tavily's endpoint. We specify `search_depth="basic"` to keep token usage and costs down.
* **Error handling**: The entire block is wrapped in `try-except` so a network issue or API limit failure doesn't crash our pipeline, returning a clean structured error block instead.

---

### B. Web Scraping (`web_scraper.py`)

When Tavily returns URLs, our system doesn't rely purely on snippets; it actually scrapes the top links to extract deep information.

```python
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
```
* **`_HEADERS`**: Many modern websites block standard Python requests libraries because their default User-Agent is `python-requests`. We spoof a real Chrome browser on Linux to bypass bot detection screens.

```python
# Tags whose text we actually want
_KEEP_TAGS = {"p", "h1", "h2", "h3", "h4", "li", "td", "th", "blockquote", "span", "a"}

# Tags to remove entirely (including children)
_STRIP_TAGS = {"script", "style", "nav", "footer", "header", "aside", "form", "noscript", "svg"}
```
* Boilerplate removal is crucial. If we dump a whole HTML page into Claude's prompt, we waste thousands of tokens on navigation menus, footers, CSS styles, and tracking scripts.

```python
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
```
* **`tag.decompose()`**: Deletes the noise tags and all of their inner children from the HTML tree.
* **`tag.get_text(separator=" ", strip=True)`**: Extracts raw text, placing a space between items and removing trailing/leading whitespaces.
* **`combined[:max_chars]`**: Truncates the clean text to `8000` characters (roughly ~2000 tokens) to safeguard against blowing through model token limits and budgets.

```python
@tool
def scrape_url(url: str) -> dict:
    """Fetch and extract clean text content from a web page."""
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
```
* **`httpx.Client`**: A modern HTTP client. We enable `follow_redirects=True` so URLs containing shortened links or SEO redirects work perfectly.
* **`resp.raise_for_status()`**: Instantly raises an exception if the web server returned an error code like `404` or `500`.

---

### C. Screenshots (`screenshot.py`)

To visually audit competitor websites, we use **Playwright**—an industrial-grade automation browser.

```python
import os
import asyncio
from pathlib import Path
from langchain_core.tools import tool
from src.config import SCREENSHOT_TIMEOUT_SEC

SCREENSHOTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports" / "screenshots"

async def _take_screenshot(url: str, output_path: str) -> dict:
    """Async helper that launches a headless browser and screenshots the page."""
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        return {
            "url": url,
            "path": "",
            "success": False,
            "error": "playwright not installed. Run: pip install playwright && playwright install chromium",
        }

    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            page = await browser.new_page(viewport={"width": 1280, "height": 800})
            await page.goto(url, wait_until="networkidle", timeout=SCREENSHOT_TIMEOUT_SEC * 1000)
            await page.screenshot(path=output_path, full_page=True)
            await browser.close()
        return {"url": url, "path": output_path, "success": True}
    except Exception as e:
        return {"url": url, "path": "", "success": False, "error": str(e)}
```
* **`async_playwright()`**: Launches Playwright in asynchronous mode.
* **`browser.new_page(viewport=...)`**: Creates a simulated window. We set a standard desktop resolution of `1280x800`.
* **`wait_until="networkidle"`**: The browser waits until there are no new network requests active for at least 500ms. This ensures dynamic React apps, charts, and loaded fonts are fully rendered before capturing!
* **`full_page=True`**: Playwright scrolls down the entire document, capturing the full page rather than just the top fold.

```python
@tool
def take_screenshot(url: str) -> dict:
    """Take a full-page screenshot of a website."""
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Create a safe filename from the URL
    safe_name = (
        url.replace("https://", "")
        .replace("http://", "")
        .replace("/", "_")
        .replace(".", "_")[:60]
    )
    output_path = str(SCREENSHOTS_DIR / f"{safe_name}.png")

    # Run the async screenshot in a new event loop if needed
    try:
        loop = asyncio.get_running_loop()
        # If we're already in an async context, create a task
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            result = loop.run_in_executor(
                pool, lambda: asyncio.run(_take_screenshot(url, output_path))
            )
            return {"url": url, "path": output_path, "success": False, "error": "async context detected"}
    except RuntimeError:
        # No event loop running — normal case
        return asyncio.run(_take_screenshot(url, output_path))
```
* **Event Loop Bridge**: Python's `asyncio` throws a `RuntimeError` if you attempt to run `asyncio.run()` while another event loop is running on the same thread. Since LangGraph is synchronous, but Playwright is asynchronous, this wrapper checks if a loop is already running and safely falls back or initiates execution.

---

## 5. The Supervisor Orchestrator (`src/agents/supervisor.py`)

The Supervisor is the brain of the network. However, invoking LLMs simply to decide "step 2 comes after step 1" is wasteful and slow. This supervisor implementation combines **deterministic rules (short-circuiting)** with **LLM reasoning**.

```python
from langchain_core.messages import SystemMessage, HumanMessage
from src.config import get_llm
from src.state import AgentState

SUPERVISOR_PROMPT = """You are a Supervisor Agent orchestrating a competitive intelligence research team..."""
```
* The prompt defines the workers and rules (e.g., search first, then visual audit, then write, then end).

```python
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

    # ── Token-Saving Short-circuit logic ────────────────────────────────
    if "search_reader" not in completed:
        return {"next_agent": "search_reader"}
    if "visual_auditor" not in completed:
        return {"next_agent": "visual_auditor"}
    if "writer" not in completed:
        return {"next_agent": "writer"}
    if state.get("critic_pass"):
        return {"next_agent": "FINISH"}
```
* **Deterministic short-circuiting**: We inspect `completed_agents` directly. If the search agent hasn't run, we immediately route to `search_reader`. No LLM needed! This completely eliminates LLM costs for standard pipeline executions, while reserving the LLM call solely for handling unexpected, ambiguous routing edge cases.

```python
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
```
* **Validation**: Checks if the LLM output is a valid node name. If it outputs garbage, it safely defaults to `"FINISH"` to prevent the system from getting stuck.

---

## 6. Specialized Worker Agents (`src/agents/`)

### A. Search & Reader (`search_reader.py`)

This worker plans search queries, fires them, and scrapes pages.

```python
def search_reader_node(state: AgentState) -> dict:
    """Execute web search and scraping for competitor intelligence."""
    llm = get_llm(temperature=0.3, max_tokens=500)
    competitor = state.get("competitor_name", "Unknown")
    focus_areas = state.get("focus_areas", ["general"])

    errors: list[str] = []
    all_search_results: list[dict] = []
    all_scraped: list[dict] = []
```

```python
    # ── Step 1: Generate search queries via LLM ─────────────────────────
    try:
        query_response = llm.invoke([
            SystemMessage(content=SEARCH_AGENT_PROMPT),
            HumanMessage(content=f"Competitor: {competitor}\nFocus areas: {', '.join(focus_areas)}"),
        ])
        import json
        raw = query_response.content.strip()
        # Handle markdown code blocks returned by LLM
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
```
* **Robust JSON Handling**: Models often wrap JSON responses in markdown fences like ` ```json ... ``` `. The code safely splits these fences away before calling `json.loads()`.
* **Fallback lists**: If query generation fails, we don't crash. We use standard queries instead!

```python
    # ── Step 2: Execute searches ────────────────────────────────────────
    for query in queries[:3]:  # Cap at 3 to control costs
        try:
            results = search_web.invoke({"query": query})
            if isinstance(results, list):
                all_search_results.extend(results)
        except Exception as e:
            errors.append(f"Search failed for '{query}': {e}")
```
* **`search_web.invoke(...)`**: In LangChain, we invoke a tool programmatically by calling `.invoke()` with its named parameters inside a dict.

```python
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
```
* **Duplicate Prevention**: We use a `seen_urls` set to ensure we never waste time scraping the exact same page twice.
* **Appends completion state**: Returns `completed_agents` with `["search_reader"]`. The LangGraph reducer will append this to the main list.

---

### B. Visual Auditor (`visual_auditor.py`)

This agent uses multi-modal vision models to audit competitor layout designs, typography, pricing panels, and positioning.

```python
def _encode_image(path: str) -> str | None:
    """Read an image file and return its base64 encoding."""
    try:
        with open(path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")
    except Exception:
        return None
```
* **Image Encoding**: API endpoints cannot receive local filesystem paths directly. The image must be base64-encoded to travel across HTTP.

```python
def visual_auditor_node(state: AgentState) -> dict:
    competitor = state.get("competitor_name", "Unknown")
    errors: list[str] = []
    screenshot_paths: list[str] = []

    # ── Step 1: Determine URLs to screenshot ────────────────────────────
    urls_to_capture = [
        f"https://www.{competitor.lower().replace(' ', '')}.com",
        f"https://www.{competitor.lower().replace(' ', '')}.com/pricing"
    ]
```
* Guesses the home page and pricing pages based on the competitor's name.

```python
    # ── Step 2: Take screenshots ────────────────────────────────────────
    for url in urls_to_capture[:2]:
        try:
            result = take_screenshot.invoke({"url": url})
            if result.get("success"):
                screenshot_paths.append(result["path"])
            else:
                errors.append(f"Screenshot failed for {url}: {result.get('error', 'unknown')}")
        except Exception as e:
            errors.append(f"Screenshot error for {url}: {e}")
```

```python
    # ── Step 3: Analyse screenshots with Claude Vision ──────────────────
    visual_analysis = ""
    if screenshot_paths:
        llm = get_llm(temperature=0.2, max_tokens=1500)

        # Build multimodal message with images
        content_parts: list[dict] = [
            {"type": "text", "text": f"Analyze these screenshots of {competitor}'s website:"},
        ]

        for path in screenshot_paths:
            b64 = _encode_image(path)
            if b64:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                })
```
* **Multimodal Structures**: For models like Claude 3.5 Sonnet that support vision, the prompt can be passed as a structured list of blocks. Some blocks are text, others are images containing `"type": "image_url"` alongside base64 data payloads.

---

### C. Writer (`writer.py`)

The Writer acts as the compiler. It reads the raw texts, visual notes, and search highlights, structuring them into a professional Markdown layout. It also handles revisions!

```python
def writer_node(state: AgentState) -> dict:
    llm = get_llm(temperature=0.3, max_tokens=4096)
    competitor = state.get("competitor_name", "Unknown")

    # Check if this is a revision pass
    critic_feedback = state.get("critic_feedback", "")
    existing_draft = state.get("draft_report", "")
```

```python
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
```
* **Dynamic Revision Routing**: If the state has `critic_feedback`, the node automatically toggles into **revision mode**. It sends the existing draft alongside the critic's list of issues, instructing the LLM to rewrite the report rather than generating a fresh one from raw notes.

```python
    # ── First draft mode ────────────────────────────────────────────────
    research_sections: list[str] = []

    # Search results & Scraped content compiler...
    # Compiles raw findings into a neat research brief (truncated to save token costs)
    research_brief = "\n\n".join(research_sections) if research_sections else "No research data available."

    response = llm.invoke([
        SystemMessage(content=WRITER_PROMPT.format(competitor=competitor, date=date_str)),
        HumanMessage(content=f"Here is all the research data gathered:\n\n{research_brief}"),
    ])

    return {
        "draft_report": response.content,
        "revision_count": 0,
        "completed_agents": ["writer"],
    }
```

---

### D. Critic (`critic.py`)

The Critic acts as the gatekeeper. It reviews the draft report against key quality requirements.

```python
CRITIC_PROMPT = """You are a strict quality reviewer...
Review the report below against these criteria:
1. Source Citations (Critical)
2. SWOT Completeness (At least 3 items)
3. Structure matches schema
...
Respond with a JSON object:
{
    "pass": true/false,
    "score": 0-100,
    "issues": ["..."],
    "summary": "..."
}"""
```

```python
def critic_node(state: AgentState) -> dict:
    llm = get_llm(temperature=0.0, max_tokens=500)
    draft = state.get("draft_report", "")
    revision_count = state.get("revision_count", 0)

    # ── Safe Escape: Max retries ────────────────────────────────────────
    if revision_count >= MAX_CRITIC_RETRIES:
        return {
            "critic_pass": True,
            "critic_feedback": "Accepted after maximum revision attempts.",
            "final_report": draft,
        }
```
* **Infinite Loop Prevention**: If the writer is struggling to satisfy a hyper-strict critic, we don't want the network spinning in loops wasting API dollars. If `revision_count` exceeds limits (e.g. `2`), the critic automatically waves the report through.

```python
    try:
        response = llm.invoke([
            SystemMessage(content=CRITIC_PROMPT),
            HumanMessage(content=f"## Report to Review:\n\n{draft}"),
        ])

        import json
        raw = response.content.strip()
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
```
* **Dynamic Gate**: If `passed` is True, `final_report` is written. If False, it is left empty, and the supervisor will route flow right back to the Writer with the populated `critic_feedback` list.

---

## 7. Graph Assembly & Routing (`src/graph.py`)

Now that we have all the nodes, let's assemble them into a workflow graph using LangGraph.

```python
from langgraph.graph import StateGraph, END
from src.state import AgentState
from src.agents.supervisor import supervisor_node
# ...
```

```python
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
```
* **Routing Helpers**: Functions that inspect the updated state and return a string corresponding to the key of the next target node.

```python
def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # ── 1. Add nodes ───────────────────────────────────────────────────
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("search_reader", search_reader_node)
    graph.add_node("visual_auditor", visual_auditor_node)
    graph.add_node("writer", writer_node)
    graph.add_node("critic", critic_node)

    # ── 2. Set entry point ──────────────────────────────────────────────
    graph.set_entry_point("supervisor")
```
* **`StateGraph(AgentState)`**: Initializes the state machine, instructing it to use `AgentState` schema to enforce dict types and apply reducers.
* **`set_entry_point`**: Sets the initial start node when the execution begins.

```python
    # ── 3. Conditional routing from supervisor ──────────────────────────
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
```
* **`add_conditional_edges`**: Defines a dynamically routed edge.
  * Argument 1: The source node (`"supervisor"`).
  * Argument 2: The router function (`_route_supervisor`).
  * Argument 3: A lookup dictionary mapping the return string of the router function to another node in the graph (or the special built-in `END` node).

```python
    # ── 4. Worker nodes return to supervisor ────────────────────────────
    graph.add_edge("search_reader", "supervisor")
    graph.add_edge("visual_auditor", "supervisor")
```
* **`add_edge(source, target)`**: Defines static transitions. Once a worker finishes, control is immediately handed back to the orchestrator to decide the next step.

```python
    # ── 5. Writer → Critic → (Writer | END) loop ────────────────────────
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
```
* **`graph.compile()`**: Performs structural validations (ensuring no disconnected nodes or missing routes exist) and compiles the graph definition into a runnable LangGraph application.

---

## 8. CLI and Execution Flow (`main.py`)

This handles user interactions, executes the compiled graph, streams progress real-time to the screen, and writes the resulting report.

```python
def main() -> None:
    # Parsing CLI args (competitor, focus areas, output location)...
```

```python
    initial_state = {
        "competitor_name": args.competitor,
        "focus_areas": focus_areas,
        "next_agent": "",
        "completed_agents": [],
        "search_results": [],
        "scraped_content": [],
        "screenshot_paths": [],
        "visual_analysis": "",
        "draft_report": "",
        "critic_feedback": "",
        "critic_pass": False,
        "revision_count": 0,
        "final_report": "",
        "errors": [],
    }

    graph = build_graph()
```
* Defines the full dictionary starting template conforming to our schema.

```python
    # Stream events to show progress
    final_state = None
    with Progress(...) as progress:
        task = progress.add_task("Initializing agents...", total=None)

        for event in graph.stream(initial_state, {"recursion_limit": 25}):
            for node_name, updates in event.items():
                # Check which node finished executing and update terminal output
                if node_name == "supervisor":
                    next_agent = updates.get("next_agent", "")
                    progress.update(task, description=f"Supervisor → routing to {next_agent}")
                # ...
                
                final_state = {**initial_state, **(final_state or {}), **updates}
```
* **`graph.stream(initial_state, {"recursion_limit": 25})`**: Runs the graph! Instead of blocking until the whole graph completes, `stream` yields key-value events immediately after each node finishes.
* **`recursion_limit`**: Protects against unexpected infinite routing loops by hard-terminating execution if it exceeds 25 steps.
* **Event Structure**: Each yielded chunk is a dictionary format `{"node_name": {state_updates_returned_by_node}}`. We merge these updates into a running `final_state` tracking dictionary.

```python
    # Save markdown reports, preview them in high-end panels using rich, and print non-fatal warnings...
```

---

## 💡 Summary of Design Patterns to Adopt

When coding your own agentic applications, copy these core best practices:
1. **Deterministic Short-Circuiting**: Use code logic rather than expensive LLM requests to make simple routing actions.
2. **Aggressive Text Stripping**: Filter and prune your scraped HTML. Large raw strings trigger massive token usage and slow responses.
3. **Structured API Schemas**: Force models to output strict JSON schemas by requesting explicit keys, parsing them safely, and building automatic fallback systems for robustness.
4. **Iterative Writer-Critic Loops**: Keep draft generation and content checking separated. Putting auditing in a dedicated critic layer creates a highly resilient quality gate.
