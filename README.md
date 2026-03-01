# Deep Equity Research System

A 26-agent, 7-phase AI-powered stock analysis system that produces institutional-quality equity research reports. Completely free to run - uses Google Gemini's free tier API, yfinance, and SEC EDGAR.

---

## What This Is

This system runs **26 AI agents** organized across **7 phases** to produce a comprehensive equity research report on any publicly traded US stock. Each agent is a specialist that receives context from prior phases and builds on it. The final output is a structured markdown report covering financial analysis, qualitative assessment, multi-method valuation, competitive intelligence, and an evidence-based academic rating.

**Cost: $0.** Google Gemini free tier (10 RPM / 250 RPD) + yfinance + SEC EDGAR = no paid services required.

---

## Architecture

```
INPUT: Ticker (e.g., AAPL)
         |
         v
+-------------------------------------------------------------+
|  DATA COLLECTION (concurrent, no LLM)                        |
|  +-- SECCollector     -> SEC EDGAR filings + XBRL financials  |
|  +-- MarketDataCollector -> yfinance price, fundamentals      |
|  +-- NewsCollector    -> DuckDuckGo recent news               |
|  +-- EarningsCollector -> Earnings history + estimates        |
+-------------------------+-----------------------------------+
                          | raw_data.json
                          v
+-------------------------------------------------------------+
|  PHASE 1: Data Organization  (4 agents)                      |
|  Agent 01: SEC Filing Analyst                                |
|  Agent 02: Earnings Call Analyst                             |
|  Agent 03: Market Data Analyst                               |
|  Agent 04: News & Sentiment Analyst                          |
|  -> phase1_combined_briefing.md                               |
+-------------------------+-----------------------------------+
                          | Phase 1 briefings
                          v
+-------------------------------------------------------------+
|  PHASE 2: Financial Analysis  (5 agents)                     |
|  Agent 05: Income Statement Analyst                          |
|  Agent 06: Balance Sheet Analyst                             |
|  Agent 07: Cash Flow Analyst                                 |
|  Agent 08: Segment Analyst                                   |
|  Agent 09: Historical Trends Analyst                         |
|  -> phase2_combined_briefing.md                               |
+-------------------------+-----------------------------------+
                          | Phase 1+2 briefings
                          v
+-------------------------------------------------------------+
|  PHASE 3: Qualitative Analysis  (4 agents)                   |
|  Agent 10: Management & Governance                           |
|  Agent 11: Competitive Moat Assessment                       |
|  Agent 12: Industry & TAM Analysis                           |
|  Agent 13: Risk Factor Assessment                            |
|  -> phase3_combined_briefing.md                               |
+-------------------------+-----------------------------------+
                          | Phase 1+2+3 briefings
                          v
+-------------------------------------------------------------+
|  PHASE 4: Valuation  (5 agents)                              |
|  Agent 14: DCF Valuation (3-scenario, 7-year)                |
|  Agent 15: Comparable Company Analysis                       |
|  Agent 16: Historical Valuation Multiples                    |
|  Agent 17: Sum-of-Parts (SOTP) Valuation                     |
|  Agent 18: Revenue Evolution Model                           |
|  -> phase4_combined_briefing.md                               |
+-------------------------+-----------------------------------+
                          | Phase 1-4 briefings
                          v
+-------------------------------------------------------------+
|  PHASE 5: Forward Intelligence  (4 agents)                   |
|  Agent 19: Sector Vulnerability Mapper                       |
|  Agent 20: Competitive Threat Tracker                        |
|  Agent 21: Scenario Progression Mapper                       |
|  Agent 22: Thesis Narrative Builder                          |
|  -> phase5_combined_briefing.md                               |
+-------------------------+-----------------------------------+
                          | All prior briefings + Phase 5 full outputs
                          v
+-------------------------------------------------------------+
|  PHASE 6: Synthesis  (3 agents)                              |
|  Agent 23: Bull Case Builder                                 |
|  Agent 24: Bear Case Builder                                 |
|  Agent 25: Final Synthesis (rating, price targets, action)   |
|  -> phase6_combined_briefing.md                               |
+-------------------------+-----------------------------------+
                          | All briefings + Phase 6 synthesis
                          v
+-------------------------------------------------------------+
|  PHASE 7: Academic Evidence Rating  (1 agent)                |
|  Agent 26: Factor-Model Based Rating                         |
|  (Value, Momentum, Quality, Accruals, ROIC/WACC,             |
|   Insider Behavior, Analyst Revisions, Moat Durability)      |
+-------------------------+-----------------------------------+
                          |
                          v
                   final_report.md
```

---

## Requirements

- Python 3.10 or higher
- A free Google Gemini API key (no credit card required)
- Internet connection

---

## Setup

### Step 1: Get a Free Gemini API Key

Visit [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey) and create a free API key. The free tier allows:
- 10 requests per minute (RPM)
- 250 requests per day (RPD)
- No credit card required

### Step 2: Install Dependencies

```bash
cd stock-analyzer
pip install -r requirements.txt
```

### Step 3: Set Your API Key

```bash
export GEMINI_API_KEY=your_key_here
```

To make this permanent, add it to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.).

### Step 4: Run the Analysis

```bash
# Analyze Apple:
python run.py AAPL

# Analyze Microsoft (parallel mode - faster):
python run.py MSFT --parallel

# Analyze NVIDIA with debug logging:
python run.py NVDA --debug
```

---

## Output

All output is saved in `output/{TICKER}_{TIMESTAMP}/`:

```
output/AAPL_20260301_120000/
+-- raw_data.json                          <- All collected market/SEC data
+-- phase1_agent01_full.md                 <- Agent 01 full output
+-- phase1_agent01_briefing.md             <- Agent 01 compressed briefing
+-- phase1_agent02_full.md
+-- phase1_agent02_briefing.md
|   ... (same pattern for all 26 agents)
+-- phase1_combined_briefing.md            <- All Phase 1 briefings combined
+-- phase2_combined_briefing.md
|   ... (through phase7_combined_briefing.md)
+-- final_report.md                        <- Complete research report
```

Each `_full.md` file contains the agent's complete structured analysis.
Each `_briefing.md` file contains the compressed 600-800 word briefing used by downstream agents.
`final_report.md` assembles all agent outputs into a single, navigable document.

---

## How Long Does It Take?

| Mode | Approx. Time |
|------|-------------|
| Sequential (default) | 45-70 minutes |
| Parallel (--parallel flag) | 25-40 minutes |

The bottleneck is Gemini's free-tier rate limit of 10 RPM. The system waits 8 seconds between calls to stay safely under the limit. With 26 agents: 26 x 8s = ~3.5 minutes of pure wait time, plus actual generation time (~20-60s per call).

If you upgrade to a paid Gemini tier, you can reduce `delay_between_requests` in `config.py` to dramatically speed up the analysis.

---

## Cost

**$0.00** - completely free with the following free services:

| Service | Purpose | Cost |
|---------|---------|------|
| Google Gemini free tier | All LLM calls | Free (250 req/day) |
| yfinance | Market data, financials | Free |
| SEC EDGAR | Filings, XBRL data | Free (public API) |
| DuckDuckGo | News search | Free (via duckduckgo-search library) |

---

## Customization

### Modify Agent Prompts
All 26 agent prompts are in `prompts.py`. Each is a function that returns a complete prompt string. To change how an agent analyzes data, edit its function. Prompts are ~50-150 lines each and very specific about output format.

### Adjust Rate Limits
In `config.py`, modify `RATE_LIMIT`:
```python
RATE_LIMIT = {
    "requests_per_minute": 8,        # Stay under 10 RPM
    "delay_between_requests": 8.0,   # Reduce if on paid tier
    ...
}
```

### Change the Model
In `config.py`:
```python
MODEL_NAME = "gemini-2.5-flash"     # Default free tier
# MODEL_NAME = "gemini-2.5-pro"     # More capable, but uses quota faster
```

### Add New Agents
1. Add the agent to `PHASE_CONFIG` in `config.py`
2. Write the prompt function in `prompts.py`
3. Call it in the appropriate `_run_phase_N()` method in `orchestrator.py`

### Adjust Scoring Weights
Each Phase 2 agent uses a scoring framework with explicit weights (e.g., Revenue Growth Quality 25%, Margin Trajectory 30%). These weights are in the prompt text in `prompts.py` - edit them to match your investment philosophy.

---

## Limitations & Caveats

1. **This is not financial advice.** The system produces research output, not investment recommendations. Always do your own due diligence.

2. **Data quality varies by company.** Large-cap US stocks (AAPL, MSFT, NVDA) have rich SEC EDGAR data. Small-caps, foreign private issuers, or recently listed companies may have sparser data.

3. **News is limited.** The DuckDuckGo news search captures publicly available snippets but not full article text or paywalled content. Agents work with what's available and flag gaps.

4. **LLM hallucination risk.** Agents are instructed to write N/A rather than fabricate data, but LLMs can still make errors. Always verify key numbers against primary sources.

5. **Free tier rate limits.** 250 requests per day means you can run approximately 9 full analyses per day (26 agents + overhead). Plan accordingly.

6. **Earnings call transcripts.** Full transcript text is not available via free APIs. The Earnings Call Analyst (Agent 02) works with EPS beat/miss data and analyst estimates rather than full transcript text.

7. **Real-time data.** yfinance data can lag by 15-30 minutes for prices. The analysis reflects data as of the collection time, not intraday.

---

## File Structure

```
stock-analyzer/
+-- run.py                 <- Entry point (use this)
+-- orchestrator.py        <- Main coordination logic
+-- prompts.py             <- All 26 agent prompts (the "brains")
+-- data_collectors.py     <- SEC, yfinance, news, earnings data
+-- llm_client.py          <- Gemini API client with rate limiting
+-- config.py              <- All configuration settings
+-- requirements.txt       <- Python dependencies
+-- README.md              <- This file
+-- output/                <- All analysis outputs saved here
    +-- {TICKER}_{DATE}/
        +-- raw_data.json
        +-- phase*_agent*_full.md
        +-- phase*_agent*_briefing.md
        +-- phase*_combined_briefing.md
        +-- final_report.md
```

---

## Troubleshooting

**"GEMINI_API_KEY environment variable not set"**
```bash
export GEMINI_API_KEY=your_key_here
```

**"429 quota error"**
You've hit the free tier limit. Wait until the minute resets (for RPM errors) or until tomorrow (for RPD errors). The system automatically retries with exponential backoff starting at 60 seconds.

**"Ticker not found in SEC company_tickers.json"**
The ticker may not be listed in SEC EDGAR (foreign companies, OTC stocks, ETFs). The system will continue with yfinance data only and note the limitation.

**Analysis seems stuck**
Each LLM call can take 20-60 seconds. With 8-second delays between calls, total wall time for 26 agents is ~45-70 minutes. This is normal. Check the terminal for progress updates.

**yfinance returns empty data**
Try a different ticker format. Some tickers need exchange suffixes (e.g., `BRK-B` not `BRKB`).
