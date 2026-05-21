"""
Visual Auditor Agent — screenshots competitor pages and analyses them visually.

Uses Playwright to capture full-page screenshots, then sends them to Claude's
vision capabilities for UI/pricing/packaging analysis.
"""

import base64
from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage
from src.config import get_llm
from src.state import AgentState
from src.tools.screenshot import take_screenshot

VISUAL_ANALYSIS_PROMPT = """You are a Visual Auditor agent analyzing competitor website screenshots.

Examine the screenshot(s) and provide a detailed analysis covering:

1. **UI/UX Design**: Layout quality, visual hierarchy, color scheme, typography
2. **Pricing Display**: How pricing is presented, any tiering strategies, free vs paid positioning
3. **Call-to-Action**: What CTAs are prominent, their placement and messaging
4. **Trust Signals**: Testimonials, logos, security badges, social proof
5. **Product Positioning**: How they frame their value proposition above the fold
6. **Notable Changes**: Anything that looks recently updated or A/B tested

Be specific and actionable. This analysis feeds into a competitive SWOT report."""


def _encode_image(path: str) -> str | None:
    """Read an image file and return its base64 encoding."""
    try:
        with open(path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")
    except Exception:
        return None


def visual_auditor_node(state: AgentState) -> dict:
    """Screenshot competitor pages and analyse them with Claude vision."""
    competitor = state.get("competitor_name", "Unknown")
    errors: list[str] = []
    screenshot_paths: list[str] = []

    # ── Step 1: Determine URLs to screenshot ────────────────────────────
    # Try the main website first, then any interesting URLs from search
    urls_to_capture = [f"https://www.{competitor.lower().replace(' ', '')}.com"]

    # Add pricing page guess
    base = urls_to_capture[0]
    urls_to_capture.append(f"{base}/pricing")

    # ── Step 2: Take screenshots ────────────────────────────────────────
    for url in urls_to_capture[:2]:  # Max 2 screenshots to save time
        try:
            result = take_screenshot.invoke({"url": url})
            if result.get("success"):
                screenshot_paths.append(result["path"])
            else:
                errors.append(f"Screenshot failed for {url}: {result.get('error', 'unknown')}")
        except Exception as e:
            errors.append(f"Screenshot error for {url}: {e}")

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

        if len(content_parts) > 1:  # We have at least one image
            try:
                response = llm.invoke([
                    SystemMessage(content=VISUAL_ANALYSIS_PROMPT),
                    HumanMessage(content=content_parts),
                ])
                visual_analysis = response.content
            except Exception as e:
                errors.append(f"Vision analysis failed: {e}")
                visual_analysis = f"Visual analysis unavailable due to error: {e}"
        else:
            visual_analysis = "No screenshots could be captured for visual analysis."
    else:
        visual_analysis = "No screenshots were captured. Screenshot tool may not be configured."

    return {
        "screenshot_paths": screenshot_paths,
        "visual_analysis": visual_analysis,
        "completed_agents": ["visual_auditor"],
        "errors": errors,
    }
