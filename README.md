# Multi-Agent Competitor Intelligence Network

A multi-agent system that orchestrates specialized AI agents to compile comprehensive market intelligence reports. Give it a competitor name and get a structured SWOT analysis backed by live web data and visual website audits.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     USER INPUT                               │
│                  "Analyze Notion"                             │
└──────────────────┬───────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────┐
│                  SUPERVISOR AGENT                            │
│          Routes tasks to specialized workers                 │
└──────┬───────────────┬───────────────────┬───────────────────┘
       │               │                   │
       ▼               ▼                   ▼
┌──────────────┐ ┌─────────────┐ ┌─────────────────────────┐
│ Search &     │ │   Visual    │ │   Writer → Critic Loop  │
│ Reader Agent │ │   Auditor   │ │                         │
│              │ │   Agent     │ │  Writer drafts report   │
│ • Tavily     │ │             │ │  Critic reviews it      │
│   Search     │ │ • Playwright│ │  Revision if needed     │
│ • BeautifulS │ │   Screenshot│ │  Max 2 revisions        │
│   Scraping   │ │ • Claude    │ │                         │
│              │ │   Vision    │ │                         │
└──────────────┘ └─────────────┘ └─────────────────────────┘
       │               │                   │
       └───────────────┴───────────────────┘
                       │
                       ▼
              ┌────────────────┐
              │  SWOT Report   │
              │  (Markdown)    │
              └────────────────┘
```

## Agent Descriptions

| Agent | Role | LangChain Components |
|-------|------|---------------------|
| **Supervisor** | Orchestrates workflow, routes tasks | LangGraph conditional edges |
| **Search & Reader** | Web search + page scraping | Tavily API, BeautifulSoup, httpx |
| **Visual Auditor** | Screenshot capture + visual analysis | Playwright, Claude Vision |
| **Writer** | Compiles SWOT intelligence report | LCEL chain with structured prompts |
| **Critic** | Quality gate with revision loop | LCEL chain, JSON validation |

## Setup

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install Playwright browsers (for screenshots)
playwright install chromium

# 4. Configure API keys
cp .env.example .env
# Edit .env with your keys
```

### Required API Keys

| Key | Where to get it | Cost |
|-----|----------------|------|
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) | ~$0.25/MTok input |
| `TAVILY_API_KEY` | [tavily.com](https://tavily.com) | Free tier: 1000 searches/mo |

## Usage

```bash
# Basic usage
python main.py "Notion"

# With specific focus areas
python main.py "Stripe" --focus "pricing,API,developer experience"

# Custom output path
python main.py "Figma" --output reports/figma_report.md
```

## Output

Reports are saved to `reports/` directory as markdown files containing:
- Executive Summary
- Company Overview
- SWOT Analysis (Strengths, Weaknesses, Opportunities, Threats)
- Product & Feature Analysis
- Pricing & Packaging
- Visual & UX Assessment
- Key Takeaways & Recommendations
- Sources

## Cost Estimate

Uses Claude 4.5 Haiku (cheapest Claude model) for all agents:
- **Typical run**: ~5-8 LLM calls → **~$0.02-0.05 per report**
- Tavily search: Free tier covers 1000 searches/month

## Project Structure

```
langlang/
├── main.py                     # CLI entry point
├── requirements.txt
├── .env.example
├── src/
│   ├── config.py               # API keys, model config, constants
│   ├── state.py                # Shared AgentState TypedDict
│   ├── graph.py                # LangGraph workflow definition
│   ├── agents/
│   │   ├── supervisor.py       # Orchestrator agent
│   │   ├── search_reader.py    # Web search + scraping agent
│   │   ├── visual_auditor.py   # Screenshot + vision analysis
│   │   ├── writer.py           # Report compilation (LCEL)
│   │   └── critic.py           # Quality review (LCEL)
│   └── tools/
│       ├── web_search.py       # Tavily search wrapper
│       ├── web_scraper.py      # httpx + BS4 scraper
│       └── screenshot.py       # Playwright screenshot tool
└── reports/                    # Generated reports
```
