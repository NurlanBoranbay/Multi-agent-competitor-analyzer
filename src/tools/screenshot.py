"""
Screenshot tool using Playwright.

Captures full-page screenshots of competitor websites for the
Visual Auditor agent to analyse with Claude's vision capabilities.
"""

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


@tool
def take_screenshot(url: str) -> dict:
    """Take a full-page screenshot of a website.

    Args:
        url: The URL to screenshot.

    Returns:
        A dict with keys: url, path, success, and optionally error.
    """
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
            # This path shouldn't be hit in normal LangGraph usage
            return {"url": url, "path": output_path, "success": False, "error": "async context detected"}
    except RuntimeError:
        # No event loop running — normal case
        return asyncio.run(_take_screenshot(url, output_path))
