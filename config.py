"""
Configuration for Deep Equity Research System.
All settings in one place - modify here to tune the system.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# API Keys
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

if not GEMINI_API_KEY:
    import warnings
    warnings.warn(
        "GEMINI_API_KEY environment variable not set. "
        "Get a free key at https://aistudio.google.com/apikey and run: "
        "export GEMINI_API_KEY=your_key_here",
        RuntimeWarning,
        stacklevel=2,
    )

# ---------------------------------------------------------------------------
# Model Settings
# ---------------------------------------------------------------------------
# gemini-2.5-flash: best free tier model (10 RPM, 250 RPD as of 2026)
MODEL_NAME = "gemini-2.5-flash"

# Temperature settings
TEMP_ANALYSIS = 0.3      # For factual financial analysis (lower = more consistent)
TEMP_CREATIVE = 0.5      # For thesis writing, narrative synthesis (slightly higher)

# Token limits per call
MAX_TOKENS_FULL = 8192   # Full analysis output
MAX_TOKENS_BRIEF = 4096  # Briefing-only calls

# ---------------------------------------------------------------------------
# Rate Limiting (Free Tier: 10 RPM, 250 RPD)
# ---------------------------------------------------------------------------
RATE_LIMIT = {
    "requests_per_minute": 8,           # Stay safely under 10 RPM
    "delay_between_requests": 8.0,      # Seconds between API calls (60s / 8 RPM ~= 7.5s, use 8s)
    "daily_request_limit": 250,         # Hard cap from Google free tier
    "daily_warn_threshold": 220,        # Warn when approaching limit
    "retry_max_attempts": 3,            # Max retries on 429 errors
    "retry_initial_delay": 60.0,        # Start at 60 seconds (free tier is strict)
    "retry_backoff_factor": 2.0,        # Exponential: 60s, 120s, 240s
}

# ---------------------------------------------------------------------------
# Output Directories
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Phase & Agent Configuration
# All 27 agents organized by phase
# ---------------------------------------------------------------------------
PHASE_CONFIG = {
    1: {
        "name": "Data Collection & Organization",
        "description": "Agents receive raw collected data and organize it for downstream analysis.",
        "agents": [
            {"id": 1,  "name": "SEC Filing Analyst",          "temp": TEMP_ANALYSIS, "max_tokens": MAX_TOKENS_FULL},
            {"id": 2,  "name": "Earnings Call Analyst",       "temp": TEMP_ANALYSIS, "max_tokens": MAX_TOKENS_FULL},
            {"id": 3,  "name": "Market Data Analyst",         "temp": TEMP_ANALYSIS, "max_tokens": MAX_TOKENS_FULL},
            {"id": 4,  "name": "News & Sentiment Analyst",    "temp": TEMP_ANALYSIS, "max_tokens": MAX_TOKENS_FULL},
        ],
    },
    2: {
        "name": "Financial Analysis",
        "description": "Deep quantitative analysis of financial statements and trends.",
        "agents": [
            {"id": 5,  "name": "Income Statement Analyst",    "temp": TEMP_ANALYSIS, "max_tokens": MAX_TOKENS_FULL},
            {"id": 6,  "name": "Balance Sheet Analyst",       "temp": TEMP_ANALYSIS, "max_tokens": MAX_TOKENS_FULL},
            {"id": 7,  "name": "Cash Flow Analyst",           "temp": TEMP_ANALYSIS, "max_tokens": MAX_TOKENS_FULL},
            {"id": 8,  "name": "Segment Analyst",             "temp": TEMP_ANALYSIS, "max_tokens": MAX_TOKENS_FULL},
            {"id": 9,  "name": "Historical Trends Analyst",   "temp": TEMP_ANALYSIS, "max_tokens": MAX_TOKENS_FULL},
        ],
    },
    3: {
        "name": "Qualitative Analysis",
        "description": "Assessment of management, moat, industry, and risk factors.",
        "agents": [
            {"id": 10, "name": "Management & Governance",     "temp": TEMP_ANALYSIS, "max_tokens": MAX_TOKENS_FULL},
            {"id": 11, "name": "Competitive Moat Assessment", "temp": TEMP_ANALYSIS, "max_tokens": MAX_TOKENS_FULL},
            {"id": 12, "name": "Industry & TAM Analysis",     "temp": TEMP_ANALYSIS, "max_tokens": MAX_TOKENS_FULL},
            {"id": 13, "name": "Risk Factor Assessment",      "temp": TEMP_ANALYSIS, "max_tokens": MAX_TOKENS_FULL},
        ],
    },
    4: {
        "name": "Valuation",
        "description": "Multi-method intrinsic value estimation.",
        "agents": [
            {"id": 14, "name": "DCF Valuation",               "temp": TEMP_ANALYSIS, "max_tokens": MAX_TOKENS_FULL},
            {"id": 15, "name": "Comparable Company Analysis", "temp": TEMP_ANALYSIS, "max_tokens": MAX_TOKENS_FULL},
            {"id": 16, "name": "Historical Valuation",        "temp": TEMP_ANALYSIS, "max_tokens": MAX_TOKENS_FULL},
            {"id": 17, "name": "Sum-of-Parts (SOTP)",         "temp": TEMP_ANALYSIS, "max_tokens": MAX_TOKENS_FULL},
            {"id": 18, "name": "Revenue Evolution Model",     "temp": TEMP_ANALYSIS, "max_tokens": MAX_TOKENS_FULL},
        ],
    },
    5: {
        "name": "Forward Intelligence",
        "description": "Scenario mapping, competitive threats, and thesis construction.",
        "agents": [
            {"id": 19, "name": "Sector Vulnerability Mapper", "temp": TEMP_ANALYSIS, "max_tokens": MAX_TOKENS_FULL},
            {"id": 20, "name": "Competitive Threat Tracker",  "temp": TEMP_ANALYSIS, "max_tokens": MAX_TOKENS_FULL},
            {"id": 21, "name": "Scenario Progression Mapper", "temp": TEMP_ANALYSIS, "max_tokens": MAX_TOKENS_FULL},
            {"id": 22, "name": "Thesis Narrative Builder",    "temp": TEMP_CREATIVE, "max_tokens": MAX_TOKENS_FULL},
        ],
    },
    6: {
        "name": "Synthesis",
        "description": "Bull/bear case construction and final synthesis report.",
        "agents": [
            {"id": 23, "name": "Bull Case Builder",           "temp": TEMP_CREATIVE, "max_tokens": MAX_TOKENS_FULL},
            {"id": 24, "name": "Bear Case Builder",           "temp": TEMP_CREATIVE, "max_tokens": MAX_TOKENS_FULL},
            {"id": 25, "name": "Final Synthesis",             "temp": TEMP_ANALYSIS, "max_tokens": MAX_TOKENS_FULL},
        ],
    },
    7: {
        "name": "Academic Evidence Rating",
        "description": "Factor-model-based rating grounded in academic finance research.",
        "agents": [
            {"id": 26, "name": "Evidence-Based Rating Agent", "temp": TEMP_ANALYSIS, "max_tokens": MAX_TOKENS_FULL},
        ],
    },
}

# Total agent count for display
TOTAL_AGENTS = sum(len(p["agents"]) for p in PHASE_CONFIG.values())

# ---------------------------------------------------------------------------
# SEC EDGAR settings
# ---------------------------------------------------------------------------
SEC_HEADERS = {
    "User-Agent": "StockAnalyzer research@example.com",
    "Accept-Encoding": "gzip, deflate",
}
SEC_BASE_URL = "https://data.sec.gov"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

# ---------------------------------------------------------------------------
# DuckDuckGo search settings (via duckduckgo-search library)
# ---------------------------------------------------------------------------
DDG_MAX_RESULTS = 8  # Results per query (library handles API interaction)

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
REQUEST_TIMEOUT = 30   # seconds for HTTP requests
YFINANCE_PERIOD_1Y = "1y"
YFINANCE_PERIOD_5Y = "5y"
YFINANCE_INTERVAL_DAILY = "1d"
YFINANCE_INTERVAL_MONTHLY = "1mo"
