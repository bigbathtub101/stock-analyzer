"""
Agent prompts for the Deep Equity Research System.

All 27 prompt functions are organized by phase.  Each function accepts
relevant context (raw data dicts or briefing strings from prior phases)
and returns a complete prompt string ready to be sent to the LLM.

IMPORTANT CONVENTIONS (applied to every prompt):
  - Every prompt instructs the agent to end its response with a
    600-800 word compressed briefing after the "---BRIEFING---" marker.
  - Agents are instructed to use specific numbers, not vague language.
  - If data is unavailable for a metric, agents note it as N/A.
  - Downstream agents depend entirely on the briefing section.
"""

import json
from typing import Any


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _fmt(data: Any, max_chars: int = 6000) -> str:
    """Serialize data to a JSON string, truncated if very large."""
    try:
        s = json.dumps(data, default=str, indent=2)
    except Exception:
        s = str(data)
    if len(s) > max_chars:
        s = s[:max_chars] + "\n... [truncated for length] ..."
    return s


_BRIEFING_FOOTER = """
---
MANDATORY FINAL SECTION:

After completing your full analysis, output the exact string "---BRIEFING---" on its own line,
then write a compressed briefing of 600-800 words that:
1. Contains ALL key numerical data points and specific figures you cited above.
2. Includes every score/rating you assigned, with the rationale in one sentence each.
3. States your conclusions and investment implication directly.
4. Is written so a downstream AI analyst can use it as a complete substitute for reading
   your full analysis -- no critical information should be missing.
5. Uses dense, efficient prose -- no filler, no repetition, no hedging without substance.

The briefing is the single most important output of your analysis.
"""


# ===========================================================================
# PHASE 1: DATA COLLECTION (Agents 1-4)
# ===========================================================================

def p1_sec_filing_analyst(ticker: str, company_name: str, sec_data: dict, market_data: dict) -> str:
    return f"""You are a senior equity research analyst specializing in SEC filings and financial statement analysis.

TASK: Organize and verify SEC EDGAR financial data for {company_name} ({ticker}).

RAW SEC DATA:
{_fmt(sec_data.get('financials', {}), 5000)}

RAW MARKET DATA (for cross-reference):
{_fmt({'income_stmt_annual': market_data.get('income_stmt_annual', {}),
       'balance_sheet_annual': market_data.get('balance_sheet_annual', {}),
       'cash_flow_annual': market_data.get('cash_flow_annual', {})}, 5000)}

INSTRUCTIONS:
Organize the data into clean, structured sections. Use the most recent 3 years of annual data.
Cross-reference SEC XBRL data with yfinance data. Where they disagree, note the discrepancy.
If data is unavailable for a specific metric, write "N/A" -- never fabricate numbers.

SECTION 1 -- INCOME STATEMENT (3 years, in $M unless stated)
Present as a table with rows: Revenue, COGS, Gross Profit, Gross Margin %, R&D, S&GA,
Other OpEx, Total OpEx, Operating Income, Operating Margin %, Interest Expense, Taxes,
Net Income, Net Margin %, Basic EPS, Diluted EPS.
Calculate year-over-year growth rates for Revenue, Gross Profit, Operating Income, Net Income, EPS.

SECTION 2 -- BALANCE SHEET (3 years, in $M)
Rows: Cash & Equivalents, Short-Term Investments, Total Cash, Accounts Receivable,
Inventory, Other Current Assets, Total Current Assets, PP&E (net), Goodwill,
Intangibles, Total Assets.
Liabilities: Accounts Payable, Short-Term Debt, Other Current Liabilities, Total Current Liabilities,
Long-Term Debt, Other Liabilities, Total Liabilities.
Equity: Common Stock, Retained Earnings, Total Stockholders' Equity.

SECTION 3 -- CASH FLOW STATEMENT (3 years, in $M)
Rows: Net Income, D&A, Stock-Based Compensation (SBC), Changes in Working Capital,
Other Operating Items, Operating Cash Flow (OCF), Capital Expenditures (CapEx),
Acquisitions, Other Investing CF, Investing Cash Flow, Dividends Paid, Share Buybacks,
Debt Issuance/(Repayment), Other Financing CF, Financing Cash Flow.
Calculate: Free Cash Flow (OCF - CapEx), SBC-Adjusted FCF (FCF - SBC), FCF Margin %.

SECTION 4 -- SHARE COUNT TRENDS
Basic and diluted share count over 3 years. YoY change in share count (dilution or buybacks).
Total SBC expense as % of revenue and as % of operating income.

SECTION 5 -- GAAP vs NON-GAAP RECONCILIATION
If non-GAAP data is available, show the bridge: GAAP EPS -> Non-GAAP EPS.
List each add-back item (SBC, amortization of acquired intangibles, restructuring, etc.).
Express non-GAAP adjustments as % of revenue.

SECTION 6 -- DATA QUALITY NOTES
List any data gaps, inconsistencies between SEC and yfinance, unusual items, or restatements.
Note which fiscal year-end is used.

{_BRIEFING_FOOTER}"""


def p1_earnings_call_analyst(ticker: str, company_name: str, earnings_data: dict, market_data: dict) -> str:
    return f"""You are a senior equity research analyst specializing in earnings call analysis and management communication.

TASK: Analyze earnings data and management communication for {company_name} ({ticker}).

EARNINGS DATA:
{_fmt(earnings_data, 4000)}

QUARTERLY FINANCIALS (for context):
{_fmt({'income_stmt_quarterly': market_data.get('income_stmt_quarterly', {}),
       'analyst_recommendations': market_data.get('analyst_recommendations', [])}, 3000)}

INSTRUCTIONS:
If full transcript text is not available, work with what you have (EPS beats/misses, revenue trends,
analyst estimate data). Note N/A for items you cannot assess.

SECTION 1 -- EARNINGS TRACK RECORD (last 4-6 quarters)
For each quarter: Report date, Revenue (actual vs estimate, beat/miss), EPS (actual vs estimate,
beat/miss in $ and %), guidance given for next quarter, guidance vs actual outcome.
Calculate overall beat rate and average beat magnitude.

SECTION 2 -- KEY MANAGEMENT THEMES (5-7 themes)
For each theme: Topic, supporting evidence from earnings data, trajectory (improving/worsening/stable).
Examples: AI adoption, margin expansion, capital allocation, competitive dynamics, cost cutting.

SECTION 3 -- GUIDANCE CREDIBILITY ASSESSMENT
Rate management guidance quality 1-5 on:
- Precision (do they give specific guidance or vague ranges?): X/5
- Accuracy (beat vs miss history): X/5
- Transparency (do they explain misses?): X/5
- Forward Visibility (do they guide more than 1 quarter out?): X/5
- Consistency (do they change narrative frequently?): X/5
Include brief rationale for each score.

SECTION 4 -- ANALYST Q&A INTELLIGENCE
Top 5 topics analysts have focused on (based on available data or inferred from news/context):
For each: the underlying concern, management's response pattern, whether the concern is growing or receding.

SECTION 5 -- NARRATIVE SHIFTS
Identify any significant changes in how management discusses the business.
Has tone become more cautious or more confident? When did any shift occur?
Any topics management appears to avoid?

SECTION 6 -- EARNINGS QUALITY FLAGS
High accruals, channel stuffing signals, aggressive revenue recognition, frequent non-GAAP adjustments,
inconsistency between net income and OCF. Flag any concerns explicitly.

{_BRIEFING_FOOTER}"""


def p1_market_data_analyst(ticker: str, company_name: str, market_data: dict) -> str:
    return f"""You are a senior equity research analyst specializing in market data, ownership structure, and valuation multiples.

TASK: Analyze market data for {company_name} ({ticker}).

CURRENT MARKET DATA:
{_fmt({'price': market_data.get('price', {}),
       'valuation': market_data.get('valuation', {}),
       'fundamentals': market_data.get('fundamentals', {}),
       'analyst': market_data.get('analyst', {}),
       'options': market_data.get('options', {})}, 4000)}

OWNERSHIP DATA:
{_fmt({'institutional_holders': market_data.get('institutional_holders', []),
       'insider_transactions': market_data.get('insider_transactions', [])}, 3000)}

INSTRUCTIONS:
Every section must use specific numbers. State the date of data where relevant.
If a metric is N/A, say so rather than omitting it.

SECTION 1 -- CURRENT VALUATION SNAPSHOT
Present all multiples in a table:
| Metric | Value | Industry Context |
P/E Trailing, P/E Forward, EV/Revenue, EV/EBITDA, P/S, P/B, PEG Ratio,
FCF Yield (FCF / Market Cap), EV/FCF.
For each multiple, indicate whether it is cheap, fair, or expensive relative to the market
and to what you know about the company's industry.

SECTION 2 -- ANALYST CONSENSUS ANALYSIS
Number of analysts covering the stock, rating distribution (Strong Buy/Buy/Hold/Sell/Strong Sell counts).
Average, median, high, and low price target. Implied upside/downside from current price.
Recent rating changes (upgrades/downgrades in last 6 months if visible from recommendations data).
Consensus trend: are estimates being revised up or down?

SECTION 3 -- OWNERSHIP & FLOAT ANALYSIS
Institutional ownership %: total and trend.
Top 10 institutional holders by position size (name and % of shares outstanding).
Insider ownership %: total holdings.
Short interest as % of float, short ratio (days to cover).
Float vs total shares outstanding analysis.

SECTION 4 -- INSIDER TRANSACTION ANALYSIS
List all insider transactions visible in the data (date, insider, title, type, shares, price).
Net insider buying or selling (last 6 and 12 months).
Notable patterns: are insiders buying at current prices? Any cluster of selling?
Context: SBC vesting vs discretionary purchases -- try to distinguish.

SECTION 5 -- PRICE PERFORMANCE CONTEXT
Current price vs 52-week range (percentile: at bottom, middle, top of range).
50-day and 200-day moving average status (above or below, by how much %).
YTD and 52-week performance.
Options flow signals if available (put/call ratio from open interest).

SECTION 6 -- TECHNICAL CONTEXT
Is the stock near support or resistance? Trend direction.
Relative valuation vs 5-year historical range for P/E and EV/EBITDA.

{_BRIEFING_FOOTER}"""


def p1_news_sentiment_analyst(ticker: str, company_name: str, news_data: dict, market_data: dict) -> str:
    return f"""You are a senior equity research analyst specializing in news analysis, sentiment tracking, and competitive intelligence.

TASK: Analyze news flow, sentiment, and competitive dynamics for {company_name} ({ticker}).

NEWS DATA:
{_fmt(news_data.get('news_items', []), 5000)}

COMPANY CONTEXT:
{_fmt({'company_info': market_data.get('company_info', {}),
       'fundamentals': market_data.get('fundamentals', {})}, 2000)}

INSTRUCTIONS:
Work with the news items provided. If news is sparse, acknowledge it and reason from what is available.
Do not fabricate news stories. Where you lack data, say N/A.

SECTION 1 -- NEWS TIMELINE (last 12 months)
Create a chronological table of the 10-15 most significant news items:
| Date | Headline | Category | Impact | Sentiment |
Categories: Earnings, Product Launch, M&A, Regulatory, Management, Analyst Action, Macro.
Impact: High/Medium/Low. Sentiment: Positive/Neutral/Negative.

SECTION 2 -- BULL THESIS (5-6 key arguments with evidence)
For each bull argument:
- The argument in one sentence
- Specific supporting evidence (numbers, dates, events)
- How strong is this argument? (Strong/Moderate/Weak)
- What would invalidate it?

SECTION 3 -- BEAR THESIS (5-6 key arguments with evidence)
For each bear argument: same structure as Section 2.

SECTION 4 -- AI DISRUPTION INTELLIGENCE
For each major business line of this company, assess:
- Is AI an opportunity, threat, or neutral for that line?
- Specific mechanism: what AI use cases either boost or threaten revenue?
- Timeline: when does this impact materialize (0-2 years, 2-5 years, 5+ years)?
- Competitive implication: which competitors are better/worse positioned on AI?
Conclude with an overall AI Disruption Score: Net Opportunity / Mixed / Net Threat.

SECTION 5 -- COMPETITIVE LANDSCAPE TABLE
| Competitor | Key Strength | Key Weakness | Market Share Trend | AI Positioning |
List 4-6 main competitors. Keep data factual and specific.

SECTION 6 -- NARRATIVE SHIFT ANALYSIS
Has the dominant narrative around this stock changed in the last 6-12 months?
What was the old narrative? What is the new narrative?
What event or data triggered the shift?
Is the new narrative priced in or still being priced in?

{_BRIEFING_FOOTER}"""


# ===========================================================================
# PHASE 2: FINANCIAL ANALYSIS (Agents 5-9)
# ===========================================================================

def p2_income_statement_analyst(ticker: str, company_name: str, phase1_briefings: str, raw_data: dict) -> str:
    return f"""You are a senior equity research analyst specializing in income statement analysis and earnings quality.

TASK: Deep-dive income statement analysis for {company_name} ({ticker}).

PHASE 1 BRIEFINGS (from prior agents):
{phase1_briefings}

SUPPLEMENTAL FINANCIAL DATA:
{_fmt({'income_stmt_annual': raw_data.get('market', {}).get('income_stmt_annual', {}),
       'income_stmt_quarterly': raw_data.get('market', {}).get('income_stmt_quarterly', {})}, 4000)}

SCORING FRAMEWORK (assign each dimension a score of 1-5, where 5 is best):
  * Revenue Growth Quality (weight 25%): consistency, source quality, replicability
  * Margin Trajectory (weight 30%): gross, operating, net margin trends
  * Earnings Quality (weight 25%): OCF vs net income, accruals, SBC impact
  * Operating Leverage (weight 20%): does revenue growth translate to faster income growth?

SECTION 1 -- REVENUE DECOMPOSITION
Break revenue into all identifiable segments/streams with:
- Absolute revenue ($M) for last 3 years
- YoY growth rate for each year
- Revenue mix % (each stream as % of total)
- Deceleration trend (is growth slowing? By how much per year?)
Revenue quality assessment: rate each stream 1-5 on: Recurring/Predictable, Customer Concentration,
Competitive Durability, Pricing Power.

SECTION 2 -- MARGIN ANALYSIS (3-year trend)
Present as table: Gross Margin, R&D as % of Revenue, S&GA as % of Revenue,
Operating Margin, EBITDA Margin, Net Margin.
For each margin: 3-year trend, YoY change, vs. peer average (state the peer benchmark).
Margin trajectory judgment: Expanding, Stable, or Contracting -- with specific evidence.

SECTION 3 -- SBC IMPACT ON EARNINGS QUALITY
SBC absolute ($M) and as % of revenue for 3 years.
SBC as % of operating income (this is critical -- high SBC masks true costs).
GAAP vs. non-GAAP EPS divergence and the SBC bridge.
Judgment: Is SBC excessive? Compare to revenue growth rate.

SECTION 4 -- CASH CONVERSION & EARNINGS QUALITY
OCF/Net Income ratio for 3 years (above 1.0 is strong, below 0.8 is a red flag).
Sloan Accruals Ratio = (Net Income - OCF) / Average Total Assets.
Working capital trends: are receivables growing faster than revenue? (DSO trend)
Inventory days trend (if applicable).

SECTION 5 -- OPERATING LEVERAGE ANALYSIS
Revenue growth vs. operating income growth -- is there positive operating leverage?
Fixed cost base estimate and its trend.
Contribution margin estimate for incremental revenue.
Key financial tension (one clear sentence identifying the biggest concern).

SECTION 6 -- PEER COMPARISON
For 3-4 named peers: Revenue Growth, Gross Margin, Operating Margin, EPS Growth, P/E.
Where does this company rank on each metric?

DIMENSIONAL SCORES:
- Revenue Growth Quality: X/5 -- [rationale]
- Margin Trajectory: X/5 -- [rationale]
- Earnings Quality: X/5 -- [rationale]
- Operating Leverage: X/5 -- [rationale]
- Weighted Composite (25/30/25/20): X.X/5

INVESTMENT IMPLICATION: [2-3 sentences on what this means for the thesis]

{_BRIEFING_FOOTER}"""


def p2_balance_sheet_analyst(ticker: str, company_name: str, phase1_briefings: str, raw_data: dict) -> str:
    return f"""You are a senior equity research analyst specializing in balance sheet analysis and financial risk assessment.

TASK: Deep-dive balance sheet analysis for {company_name} ({ticker}).

PHASE 1 BRIEFINGS:
{phase1_briefings}

BALANCE SHEET DATA:
{_fmt({'balance_sheet_annual': raw_data.get('market', {}).get('balance_sheet_annual', {}),
       'balance_sheet_quarterly': raw_data.get('market', {}).get('balance_sheet_quarterly', {})}, 4000)}

SCORING FRAMEWORK (1-5 each, 5 is best):
  * Liquidity (weight 25%)
  * Leverage Risk (weight 25%): lower debt = higher score
  * Asset Quality (weight 25%)
  * Capital Efficiency (weight 25%)

SECTION 1 -- LIQUIDITY ANALYSIS
Current Ratio (Current Assets / Current Liabilities) -- 3-year trend.
Quick Ratio (Cash + ST Investments + Receivables) / Current Liabilities -- 3-year trend.
Cash Ratio. Days Cash on Hand.
Total liquidity (cash + short-term investments + undrawn credit facilities if known).
Judgment: is liquidity adequate? Any liquidity risk in the next 12-24 months?

SECTION 2 -- DEBT & LEVERAGE ANALYSIS
Short-term debt, long-term debt, total debt -- 3-year trend.
Net Debt = Total Debt - Total Cash (negative = net cash position, which is positive).
Net Debt / EBITDA ratio -- 3-year trend.
Net Debt / Equity. Interest Coverage Ratio (EBIT / Interest Expense).
Debt maturity schedule (if visible from filings). Near-term maturities are a risk.
Judgment: is leverage sustainable? What stress test would break it?

SECTION 3 -- ASSET QUALITY
Goodwill as % of total assets -- is the balance sheet loaded with acquisition premium?
Goodwill impairment risk assessment.
PP&E as % of total assets -- capital intensity.
Receivables quality: DSO trend. Are receivables growing faster than revenue (bad sign)?
Inventory trends (if applicable): inventory days, write-down risk.

SECTION 4 -- CAPITAL EFFICIENCY
Return on Equity (ROE) = Net Income / Avg Stockholders' Equity -- 3-year trend.
Return on Assets (ROA) = Net Income / Avg Total Assets.
Return on Invested Capital (ROIC) = NOPAT / (Total Debt + Equity - Cash).
Book Value per Share trend -- is it growing? Buyback effect?
Asset turnover ratio trend.

SECTION 5 -- DILUTION TRAJECTORY
Share count trend (3 years): basic, diluted.
Net dilution rate (new shares issued via SBC minus buybacks, as % per year).
Buyback analysis: are buybacks exceeding SBC issuance (accretive) or less (dilutive net)?
At current dilution rate, projected share count in 3 years.

DIMENSIONAL SCORES:
- Liquidity: X/5 -- [rationale]
- Leverage Risk: X/5 -- [rationale]
- Asset Quality: X/5 -- [rationale]
- Capital Efficiency: X/5 -- [rationale]
- Weighted Composite: X.X/5

INVESTMENT IMPLICATION: [2-3 sentences]

{_BRIEFING_FOOTER}"""


def p2_cash_flow_analyst(ticker: str, company_name: str, phase1_briefings: str, raw_data: dict) -> str:
    return f"""You are a senior equity research analyst specializing in cash flow analysis and capital allocation.

TASK: Deep-dive cash flow analysis for {company_name} ({ticker}).

PHASE 1 BRIEFINGS:
{phase1_briefings}

CASH FLOW DATA:
{_fmt({'cash_flow_annual': raw_data.get('market', {}).get('cash_flow_annual', {}),
       'cash_flow_quarterly': raw_data.get('market', {}).get('cash_flow_quarterly', {})}, 4000)}

SCORING FRAMEWORK (1-5 each, 5 is best):
  * OCF Quality (25%): is OCF consistently above net income?
  * FCF Generation (25%): absolute FCF and FCF margin trend
  * Capex Efficiency (25%): is capex generating returns?
  * Cash Deployment Quality (25%): is management allocating capital wisely?

SECTION 1 -- OPERATING CASH FLOW QUALITY (3 years)
OCF absolute ($M) and YoY growth.
OCF Margin (OCF / Revenue) -- trend and vs. peers.
OCF / Net Income ratio -- values above 1.0 indicate earnings quality.
Key working capital drivers: was OCF boosted by working capital timing? (analyze components)
D&A as % of revenue and as % of capex (D&A/Capex > 1.0 means assets depreciating faster than replaced).

SECTION 2 -- FREE CASH FLOW ANALYSIS (3 years)
Standard FCF = OCF - CapEx. FCF Margin = FCF / Revenue.
SBC-Adjusted FCF = FCF - SBC. SBC-Adjusted FCF Margin.
FCF conversion rate = FCF / Net Income.
FCF yield = FCF / Market Cap (current). Is the stock cheap on FCF yield?
FCF trend: expanding, stable, or contracting?

SECTION 3 -- CAPEX ANALYSIS
CapEx absolute ($M) and as % of revenue.
CapEx / D&A ratio (>1 = growing asset base; <1 = under-investing or asset-light).
Maintenance vs. Growth CapEx split (estimate if not disclosed).
Capex efficiency: is FCF growing despite high capex? Or is capex a margin drag?

SECTION 4 -- CASH DEPLOYMENT PRIORITIES
Rank management's actual cash deployment for last 3 years:
1. Capital expenditures: $Xm (X% of OCF)
2. M&A/acquisitions: $Xm (X% of OCF)
3. Share buybacks: $Xm (X% of OCF)
4. Dividends: $Xm (X% of OCF)
5. Debt repayment: $Xm (X% of OCF)
Judgment: Is capital allocation value-accretive? Are buybacks at reasonable valuations?

SECTION 5 -- FCF FORWARD PROJECTION
Based on trailing trends, estimate FCF for next 2 years (base case):
State assumptions clearly (revenue growth, margin, capex/rev).
FCF yield at current market cap for each scenario.

DIMENSIONAL SCORES:
- OCF Quality: X/5 -- [rationale]
- FCF Generation: X/5 -- [rationale]
- Capex Efficiency: X/5 -- [rationale]
- Cash Deployment Quality: X/5 -- [rationale]
- Weighted Composite: X.X/5

INVESTMENT IMPLICATION: [2-3 sentences]

{_BRIEFING_FOOTER}"""


def p2_segment_analyst(ticker: str, company_name: str, phase1_briefings: str, raw_data: dict) -> str:
    return f"""You are a senior equity research analyst specializing in business segment analysis and revenue mix assessment.

TASK: Segment and revenue mix analysis for {company_name} ({ticker}).

PHASE 1 BRIEFINGS:
{phase1_briefings}

FINANCIAL DATA:
{_fmt({'income_stmt_annual': raw_data.get('market', {}).get('income_stmt_annual', {}),
       'fundamentals': raw_data.get('market', {}).get('fundamentals', {})}, 3000)}

SCORING FRAMEWORK (1-5 each, 5 is best):
  * Diversification (25%): revenue across multiple durable streams
  * Growth Profile (25%): overall growth and high-growth segment contribution
  * Margin Mix (25%): are high-margin segments growing as % of revenue?
  * Concentration Risk (25%): no single customer, product, or geography >30% of revenue

SECTION 1 -- SEGMENT REVENUE ANALYSIS
For each identifiable business segment/revenue stream:
| Segment | FY-2 Revenue | FY-1 Revenue | FY0 Revenue | 2-yr CAGR | Mix % FY0 |
If segment data is not broken out in available data, infer from company disclosures and context.
Note which segments are disclosed vs. inferred.

SECTION 2 -- SEGMENT PROFITABILITY
For each segment:
| Segment | Est. Gross Margin | Est. Operating Margin | vs. Company Average | Trend |
If segment margins are not disclosed, estimate using industry comparables and context.
Flag which segments are margin accretive vs. dilutive.

SECTION 3 -- REVENUE MIX SHIFT
Is the revenue mix shifting toward higher-margin, higher-growth segments?
Quantify: what % of revenue did the highest-margin segment contribute 3 years ago vs. today?
Is this mix shift accelerating or stalling?
Project mix in 3 years if current trends continue.

SECTION 4 -- SUBSCRIPTION / RECURRING REVENUE ANALYSIS
What % of revenue is recurring (subscriptions, maintenance, SaaS)?
What % is transactional or one-time?
Trend in recurring revenue %: this is the most important mix shift to track.
Net Revenue Retention (NRR) or equivalent metric if available.

SECTION 5 -- PRODUCT/GEOGRAPHIC CONCENTRATION RISK
Top 3 customers as % of revenue (if disclosed). Single-customer concentration risk?
Geographic breakdown: domestic vs. international. FX exposure?
Product concentration: is there a hero product driving most revenue? What replaces it?

DIMENSIONAL SCORES:
- Diversification: X/5 -- [rationale]
- Growth Profile: X/5 -- [rationale]
- Margin Mix: X/5 -- [rationale]
- Concentration Risk: X/5 -- [rationale]
- Weighted Composite: X.X/5

INVESTMENT IMPLICATION: [2-3 sentences]

{_BRIEFING_FOOTER}"""


def p2_historical_trends_analyst(ticker: str, company_name: str, phase1_briefings: str, raw_data: dict) -> str:
    return f"""You are a senior equity research analyst specializing in historical financial trends and capital return analysis.

TASK: Historical trends and long-term trajectory analysis for {company_name} ({ticker}).

PHASE 1 BRIEFINGS:
{phase1_briefings}

MULTI-YEAR FINANCIAL DATA:
{_fmt({'income_stmt_annual': raw_data.get('market', {}).get('income_stmt_annual', {}),
       'balance_sheet_annual': raw_data.get('market', {}).get('balance_sheet_annual', {}),
       'cash_flow_annual': raw_data.get('market', {}).get('cash_flow_annual', {})}, 4000)}

SCORING FRAMEWORK (1-5 each, 5 is best):
  * Growth Consistency (25%): no negative growth years, low variance
  * Margin Trajectory (25%): long-term margin expansion
  * Cyclicality (25%): low cyclicality = higher score
  * Capital Returns (25%): sustained high ROIC, disciplined buybacks/dividends

SECTION 1 -- REVENUE & EPS HISTORY (as many years as available, ideally 5-7)
| Year | Revenue ($M) | YoY Growth | Operating Margin | EPS (diluted) | EPS Growth |
Calculate: Revenue 5-year CAGR, EPS 5-year CAGR.
Standard deviation of annual revenue growth (lower = more consistent).

SECTION 2 -- KEY INFLECTION POINTS
Identify 3-5 major turning points in the company's financial history.
For each: Year, Event (product launch, acquisition, management change, economic shock),
Impact on Revenue, Impact on Margins, Lasting effect (temporary or structural change)?

SECTION 3 -- ROIC ANALYSIS
Calculate or estimate ROIC for each of the last 3-5 years.
Estimate WACC (use: risk-free rate ~4.5%, equity risk premium ~5.5%, use beta from market data).
ROIC vs. WACC spread: is the company creating or destroying economic value?
ROIC trend: is it improving, stable, or declining?

SECTION 4 -- GUIDANCE BEAT/MISS PATTERNS
Over the last 4-8 quarters, score: beat / miss / in-line for Revenue and EPS.
Calculate beat rate (%) and average beat magnitude for each metric.
Is the company a conservative or aggressive guide-and-raise operator?
Any systematic pattern (over-promise early in year, reduce later)?

SECTION 5 -- CAPITAL RETURN HISTORY
Dividend history (if any): initiation date, current yield, growth rate.
Buyback history: total $ returned, shares reduced over 5 years as % of original count.
Total Shareholder Return (TSR) over 1, 3, and 5 years vs. S&P 500.

DIMENSIONAL SCORES:
- Growth Consistency: X/5 -- [rationale]
- Margin Trajectory: X/5 -- [rationale]
- Cyclicality: X/5 -- [rationale]
- Capital Returns: X/5 -- [rationale]
- Weighted Composite: X.X/5

KEY FINANCIAL TENSION: [One sentence identifying the single biggest financial concern or contradiction]

INVESTMENT IMPLICATION: [2-3 sentences]

{_BRIEFING_FOOTER}"""


# ===========================================================================
# PHASE 3: QUALITATIVE ANALYSIS (Agents 10-13)
# ===========================================================================

def p3_management_governance(ticker: str, company_name: str, phase1_2_briefings: str, raw_data: dict) -> str:
    return f"""You are a senior equity research analyst specializing in management quality assessment and corporate governance.

TASK: Management and governance deep-dive for {company_name} ({ticker}).

PHASE 1-2 BRIEFINGS:
{phase1_2_briefings}

INSIDER & OWNERSHIP DATA:
{_fmt({'insider_transactions': raw_data.get('market', {}).get('insider_transactions', []),
       'institutional_holders': raw_data.get('market', {}).get('institutional_holders', []),
       'company_info': raw_data.get('market', {}).get('company_info', {})}, 3000)}

SCORING FRAMEWORK (1-5 each, 5 is best):
  * CEO Quality (weight 35%): track record, strategic clarity, capital allocation
  * Team Stability (weight 20%): tenure, low turnover, depth of bench
  * Insider Alignment (weight 20%): ownership, buying behavior
  * Governance Structure (weight 25%): board independence, shareholder rights

SECTION 1 -- CEO ASSESSMENT
Name, tenure (years in role), background (prior companies, functional expertise).
Key strategic decisions made and their outcomes (use financial data to grade them):
  - M&A track record: did acquisitions create or destroy value?
  - Capital allocation: buybacks at good prices? Capex ROI?
  - Operational track record: margin improvement, revenue growth vs. peers?
Capital allocation grade: A (excellent) / B (good) / C (average) / D (poor) / F (destructive).
Communication quality: transparent, honest, or promotional?

SECTION 2 -- FOUNDER vs PROFESSIONAL MANAGEMENT
Is this founder-led? If so: does the founder still control vision? What happens post-founder?
If professional management: how long has the team been together? Any recent C-suite departures?
Key C-suite members (CEO, CFO, COO, heads of business): tenure, background, credibility.

SECTION 3 -- INSIDER TRANSACTION ANALYSIS
List all insider transactions from data (date, name, title, transaction type, shares, price, value).
Net insider buying/(selling) in last 6 months and 12 months.
Classify: was selling plan-based (10b5-1) or open market? Open-market purchases are most bullish.
Conclusion: are insiders net buyers at current prices? What does this signal?

SECTION 4 -- OWNERSHIP & GOVERNANCE STRUCTURE
Share class structure: single-class or dual/multi-class? (Dual class = weaker shareholder rights)
Board composition: total directors, independent directors %, board tenure (long = entrenchment risk).
CEO/Chairman separation: combined or separate (separate is better for oversight).
Executive compensation: is pay aligned with shareholder returns? Excessive base or severance?
SBC policy: is total annual SBC excessive vs. revenue and FCF?

SECTION 5 -- STRATEGIC VISION ASSESSMENT
Is there a clear, consistent 3-5 year strategic vision?
Key initiatives underway and their progress.
Where does management think the company will be in 5 years?
Risk: is management chasing trends or executing a durable strategy?

DIMENSIONAL SCORES:
- CEO Quality: X/5 -- [rationale]
- Team Stability: X/5 -- [rationale]
- Insider Alignment: X/5 -- [rationale]
- Governance Structure: X/5 -- [rationale]
- Weighted Composite: X.X/5

INVESTMENT IMPLICATION: [2-3 sentences]

{_BRIEFING_FOOTER}"""


def p3_competitive_moat_assessment(ticker: str, company_name: str, phase1_2_briefings: str, raw_data: dict) -> str:
    return f"""You are a senior equity research analyst specializing in competitive moat analysis using Morningstar's five moat sources framework.

TASK: Competitive moat assessment for {company_name} ({ticker}).

PHASE 1-2 BRIEFINGS:
{phase1_2_briefings}

COMPANY CONTEXT:
{_fmt({'company_info': raw_data.get('market', {}).get('company_info', {}),
       'fundamentals': raw_data.get('market', {}).get('fundamentals', {})}, 2000)}

SCORING FRAMEWORK:
For each of the five moat sources, assign a score of 0-3:
  0 = Not present or very weak
  1 = Marginal (provides some advantage but not durable)
  2 = Meaningful (real competitive advantage, medium durability)
  3 = Strong (hard to replicate, highly durable)
Total score: 0-15
  0-3 = No Moat | 4-7 = Narrow Moat | 8-11 = Wide Moat | 12-15 = Very Wide Moat

SECTION 1 -- NETWORK EFFECTS (score: X/3)
Definition: Does the product/service become more valuable as more users join?
Evidence: [specific examples with data]
Durability: [how defensible is this network effect?]
AI Disruption Overlay: Does AI strengthen or weaken this network effect? How?

SECTION 2 -- SWITCHING COSTS (score: X/3)
Definition: How painful is it for customers to switch to a competitor?
Evidence: [contracts, data lock-in, integration depth, customer retention rate, NRR]
Durability: [can competitors reduce switching costs with new technology?]
AI Disruption Overlay: Does AI make switching easier (weakening moat) or harder?

SECTION 3 -- COST ADVANTAGES (score: X/3)
Definition: Can the company produce at materially lower cost than competitors?
Sources: scale economics, proprietary process, unique resource access, location.
Evidence: [gross margin vs. peers, operating leverage data]
AI Disruption Overlay: Does AI create new cost advantages or erode existing ones?

SECTION 4 -- INTANGIBLE ASSETS (score: X/3)
Definition: Patents, brand, regulatory licenses, proprietary data.
Evidence: [patent portfolio size, brand value indicators, regulatory moats, data assets]
Durability: [patent cliff risk, brand erosion risk]
AI Disruption Overlay: Does AI commoditize these intangibles?

SECTION 5 -- EFFICIENT SCALE (score: X/3)
Definition: Does the company operate in a market that can only support a few players?
Evidence: [market size, number of viable competitors, returns on capital vs. cost of capital]
Durability: [what would it take to attract a new entrant?]
AI Disruption Overlay: Does AI lower barriers to entry in this space?

SECTION 6 -- MOAT TREND
Overall moat trend: Widening / Stable / Narrowing
Evidence for the trend: [3-4 specific data points or events]
Moat durability horizon: how long before competitive advantage meaningfully erodes?

SECTION 7 -- AI DISRUPTION ASSESSMENT BY BUSINESS LINE
| Business Line | AI Role | Opportunity or Threat | Timeline | Net Impact |
For each major business line or revenue stream.
Overall AI Disruption Score: Net Opportunity / Mixed / Net Threat.

TOTAL MOAT SCORE: X/15 -> Classification: [No/Narrow/Wide/Very Wide]

INVESTMENT IMPLICATION: [2-3 sentences on how the moat (or lack thereof) affects the investment case]

{_BRIEFING_FOOTER}"""


def p3_industry_tam_analysis(ticker: str, company_name: str, phase1_2_briefings: str, raw_data: dict) -> str:
    return f"""You are a senior equity research analyst specializing in industry analysis, TAM sizing, and secular trend assessment.

TASK: Industry and TAM analysis for {company_name} ({ticker}).

PHASE 1-2 BRIEFINGS:
{phase1_2_briefings}

COMPANY CONTEXT:
{_fmt({'company_info': raw_data.get('market', {}).get('company_info', {}),
       'valuation': raw_data.get('market', {}).get('valuation', {}),
       'fundamentals': raw_data.get('market', {}).get('fundamentals', {})}, 2000)}

SCORING FRAMEWORK (1-5 each, 5 is best):
  * Market Growth (30%): overall industry growth rate
  * Competitive Intensity (20%): fewer, weaker competitors = higher score
  * Secular Tailwinds (30%): structural growth drivers
  * Disruption Risk (20%): lower disruption risk = higher score

SECTION 1 -- TAM SIZING BY BUSINESS LINE
For each major business segment:
| Segment | Current TAM ($B) | TAM Growth Rate | Company Revenue | Market Share | Share Trend |
Where TAM data is not publicly available, estimate using reasonable proxies (revenue of top 5 players,
industry reports, analyst consensus). Flag estimates as such.
Overall company TAM (sum of addressable markets). What % of TAM is currently captured?

SECTION 2 -- ADOPTION CURVE POSITIONING
For each business segment, classify on the adoption curve:
Innovators (0-2.5%) -> Early Adopters (2.5-16%) -> Early Majority (16-50%) -> Late Majority (50-84%) -> Maturity (>84%)
What does the company's growth rate imply about adoption stage?
Transition risk: what happens when a segment moves from Growth to Maturity?

SECTION 3 -- PORTER'S FIVE FORCES (one paragraph each)
For the company's primary market:
1. Threat of New Entrants: capital requirements, regulatory barriers, economies of scale.
2. Bargaining Power of Buyers: customer concentration, switching costs, alternatives.
3. Bargaining Power of Suppliers: supplier concentration, input substitution, supplier switching costs.
4. Threat of Substitutes: direct and indirect substitute products, price-performance trajectory.
5. Industry Rivalry: number of competitors, capacity utilization, differentiation, pricing dynamics.
Overall Porter's Assessment: Favorable / Neutral / Unfavorable for this company.

SECTION 4 -- SECULAR TAILWINDS & HEADWINDS
Top 3-4 secular tailwinds (structural forces that grow the market regardless of cycle):
For each: mechanism, magnitude estimate, timeline.
Top 2-3 secular headwinds (structural forces that challenge the industry):
For each: mechanism, magnitude estimate, timeline.

SECTION 5 -- AI DISRUPTION TAXONOMY (critical section)
For each major business line, classify the AI impact:
(a) AI Within Product: AI enhances the product, making it more valuable (net positive)
(b) AI Bypassing Product: AI agents or copilots can replace the product entirely (net negative)
(c) AI Compressing Seats: AI reduces the headcount that uses the product (revenue per customer drops)
Assess probability (Low/Medium/High) and timeline (0-2yr / 2-5yr / 5+yr) for each.

DIMENSIONAL SCORES:
- Market Growth: X/5 -- [rationale]
- Competitive Intensity: X/5 -- [rationale]
- Secular Tailwinds: X/5 -- [rationale]
- Disruption Risk: X/5 -- [rationale]
- Weighted Composite: X.X/5

INVESTMENT IMPLICATION: [2-3 sentences]

{_BRIEFING_FOOTER}"""


def p3_risk_factor_assessment(ticker: str, company_name: str, phase1_2_briefings: str, raw_data: dict) -> str:
    return f"""You are a senior equity research analyst specializing in risk identification, quantification, and monitoring.

TASK: Comprehensive risk assessment for {company_name} ({ticker}).

PHASE 1-2 BRIEFINGS:
{phase1_2_briefings}

COMPANY CONTEXT:
{_fmt({'company_info': raw_data.get('market', {}).get('company_info', {}),
       'fundamentals': raw_data.get('market', {}).get('fundamentals', {}),
       'valuation': raw_data.get('market', {}).get('valuation', {})}, 2000)}

SCORING: Assign overall risk score 1-5 where 5 = highest risk, 1 = lowest risk.

SECTION 1 -- COMPREHENSIVE RISK MATRIX
Present 15-20 risks in a table:
| # | Risk | Category | Probability | Impact | Timeframe | Risk Score | Monitoring Signal |
Probability: L=Low, M=Medium, H=High
Impact: L=Low, M=Medium, H=High (on stock price, 1yr timeframe)
Timeframe: <1yr, 1-3yr, 3-5yr
Risk Score: P x I (HH=9, HM/MH=6, MM/HL/LH=4, ML/LM=2, LL=1)
Categories: Financial, Operational, Regulatory/Legal, Market, AI Disruption.

Prioritized top risks by Risk Score descending.

SECTION 2 -- DEEP DIVE: TOP 3 RISKS (by probability x impact)
For each of the 3 highest-scoring risks:
Risk Name: [Name]
Description: [Specific, detailed description -- not generic]
Probability Assessment: [Why this probability? What evidence?]
Impact Quantification: [In $ or % stock price impact if the risk materializes]
Management Mitigation: [What is management doing about it? Is it adequate?]
Early Warning Trigger: [Specific, measurable signal to watch for that indicates this risk is materializing]
Historical Precedent: [Has this type of risk hurt comparable companies? Examples?]

SECTION 3 -- FINANCIAL RISK ANALYSIS
Debt covenant risk (if applicable).
Liquidity risk under stress scenario (revenue -20%, margins -5%).
Currency/FX risk exposure.
Customer concentration risk.
Supply chain risk.

SECTION 4 -- REGULATORY & LEGAL RISK
Any pending litigation or regulatory investigations.
Industry-specific regulatory risk (privacy laws, antitrust, sector regulation).
Geographic expansion regulatory risk.
Data privacy and security risk (GDPR, CCPA, breach risk).

SECTION 5 -- AI DISRUPTION RISK (most forward-looking risk category)
Identify the 2-3 most plausible AI disruption scenarios specific to this company.
For each: mechanism, timeline, probability, estimated revenue impact.
What would need to happen in AI for the bull/bear case to play out?

SECTION 6 -- RISK TREND
Are risks generally increasing or decreasing vs. 12 months ago?
Any new risks that emerged recently?
Any risks that have been largely resolved?

OVERALL RISK SCORE: X/5 -- [2-sentence justification]

INVESTMENT IMPLICATION: [2-3 sentences]

{_BRIEFING_FOOTER}"""


# ===========================================================================
# PHASE 4: VALUATION (Agents 14-18)
# ===========================================================================

def p4_dcf_valuation(ticker: str, company_name: str, phase1_3_briefings: str, raw_data: dict) -> str:
    price = raw_data.get('market', {}).get('price', {}).get('current', 'N/A')
    market_cap = raw_data.get('market', {}).get('valuation', {}).get('market_cap', 'N/A')
    return f"""You are a senior equity research analyst specializing in DCF valuation and intrinsic value analysis.

TASK: Discounted Cash Flow valuation for {company_name} ({ticker}).
Current Price: ${price} | Market Cap: ${market_cap:,}

PHASE 1-3 BRIEFINGS (contains financial analysis and qualitative assessment):
{phase1_3_briefings}

SECTION 1 -- WACC ESTIMATION
Risk-Free Rate: 4.5% (10-year Treasury)
Equity Risk Premium: 5.5% (Damodaran estimate)
Beta: [use from market data, note if adjusting toward 1.0 for mean reversion]
Size Premium: [apply small cap premium if applicable]
After-Tax Cost of Debt: [from financial data]
Debt/Capital and Equity/Capital weights: [from balance sheet]
WACC Result: X.X%

SECTION 2 -- THREE-SCENARIO DCF (7-year projection)
Build three scenarios with explicit annual projections:

BULL CASE (probability: X%):
Year 0-7 Revenue ($M), Revenue Growth %, Operating Margin %, CapEx/Revenue %,
D&A/Revenue %, Change in Working Capital as % of Revenue change, Tax Rate %,
NOPAT, FCF, SBC-Adjusted FCF.
Terminal Growth Rate: X%. Terminal Value. PV of Terminal Value.
PV of FCF years 1-7. Enterprise Value. Net Cash/(Debt). Equity Value. Per Share Value: $X.

BASE CASE (probability: X%): [same structure]
BEAR CASE (probability: X%): [same structure]

SECTION 3 -- PROBABILITY-WEIGHTED VALUE
Weighted Fair Value = (Bull % x Bull/Share) + (Base % x Base/Share) + (Bear % x Bear/Share).
Upside from current price (to weighted value): X%
Implied margin of safety at current price: X%

SECTION 4 -- SENSITIVITY ANALYSIS
Build a 5x5 sensitivity matrix: WACC (rows) vs. Terminal Growth Rate (columns).
WACC range: +/-2% from base. TGR range: 1%-4%.
Show per-share intrinsic value in each cell.
Highlight the cells where the stock is cheap vs. current price.

SECTION 5 -- REVERSE DCF
At current stock price, what FCF growth rate is implied?
Calculate what FCF margin the market is pricing in by Year 7.
Is this assumption reasonable, too optimistic, or too pessimistic?
"The market is pricing in: [specific narrative about what must be true for current price to be fair]"

VALUATION SCORE: X/5 (5 = highly attractive, 1 = significantly overvalued)

INVESTMENT IMPLICATION: [2-3 sentences: what does the DCF say about risk/reward at current price?]

{_BRIEFING_FOOTER}"""


def p4_comparable_company(ticker: str, company_name: str, phase1_3_briefings: str, raw_data: dict) -> str:
    price = raw_data.get('market', {}).get('price', {}).get('current', 'N/A')
    return f"""You are a senior equity research analyst specializing in comparable company analysis (trading comps).

TASK: Comparable company analysis (trading comps) for {company_name} ({ticker}).
Current Price: ${price}

PHASE 1-3 BRIEFINGS:
{phase1_3_briefings}

SECTION 1 -- PEER GROUP SELECTION
Select 5-8 peer companies. For each, justify its inclusion in the peer group:
- Business model similarity
- Margin profile similarity
- Growth rate similarity
- Relevant differences that require multiple adjustment

SECTION 2 -- PEER COMPARISON TABLE
Build a comprehensive peer comparison table:
| Company | Ticker | Mkt Cap ($B) | Revenue ($M) | Rev Growth | Gross Margin | Op Margin | Net Margin |
| EV/Rev | EV/EBITDA | P/E Trailing | P/E Forward | PEG | FCF Yield |

For {ticker} vs. each peer, calculate: premium or discount on each multiple.
Quartile ranking of {ticker} in the peer group for each metric.

SECTION 3 -- GROWTH-ADJUSTED MULTIPLE ANALYSIS
PEG ratio for each peer (P/E / EPS growth rate). Is {ticker} cheap or expensive on PEG?
EV/EBITDA / EBITDA Growth -- growth-adjusted enterprise multiple.
For peers with negative earnings, use EV/Revenue or EV/Gross Profit.
Identify which peers are most comparable on risk/growth/margin profile.

SECTION 4 -- SEGMENT-SPECIFIC PEER ANALYSIS
For each major business segment of {company_name}, identify the most relevant pure-play peer.
What multiple does that pure-play trade at?
Apply that multiple to {ticker}'s segment revenue/earnings.
This provides a "sum of parts" comps-based valuation (reconcile with SOTP agent).

SECTION 5 -- IMPLIED FAIR VALUE FROM COMPS
Using peer median multiples:
| Method | Peer Median Multiple | Applied to {ticker} | Implied Share Price (Bear/Base/Bull) |
EV/EBITDA method, P/E method, EV/Revenue method.
Blended comps fair value: Bear $X / Base $X / Bull $X.
Premium/discount justification: why should {ticker} trade at a premium or discount to peers?

VALUATION SCORE: X/5

INVESTMENT IMPLICATION: [2-3 sentences]

{_BRIEFING_FOOTER}"""


def p4_historical_valuation(ticker: str, company_name: str, phase1_3_briefings: str, raw_data: dict) -> str:
    price = raw_data.get('market', {}).get('price', {}).get('current', 'N/A')
    return f"""You are a senior equity research analyst specializing in historical valuation analysis and mean-reversion frameworks.

TASK: Historical valuation analysis for {company_name} ({ticker}).
Current Price: ${price}

PHASE 1-3 BRIEFINGS:
{phase1_3_briefings}

SECTION 1 -- 5-YEAR HISTORICAL MULTIPLE RANGES
For each valuation multiple, show: Minimum, 25th percentile, Median, 75th percentile, Maximum.
Multiples to cover: P/E Trailing, P/E Forward, EV/Revenue, EV/EBITDA, P/OCF, P/FCF, P/S.

SECTION 2 -- CURRENT PERCENTILE RANKING
For each multiple, where does the current value rank vs. 5-year history?
| Multiple | Current | Min | Median | Max | Current Percentile |
"Currently cheap" if percentile < 25%. "Currently rich" if percentile > 75%.
Overall assessment: is the stock historically cheap, fairly valued, or expensive?

SECTION 3 -- HISTORICAL MULTIPLE DRIVERS
What caused the multiple to expand or contract at historical extremes?
- Peak multiple: what narrative/conditions justified it?
- Trough multiple: what fears drove it there?
- Current multiple: does the current story justify the current valuation?
Mean reversion analysis: if multiples revert to median, what is the implied stock price?

SECTION 4 -- IMPLIED FAIR VALUE FROM HISTORICAL MULTIPLES
| Multiple Method | 25th Percentile Value | Median Value | 75th Percentile Value |
Apply each historical percentile to current/forward metrics.
Blended historical fair value: Bear (25th pct) $X / Base (Median) $X / Bull (75th pct) $X.

SECTION 5 -- VALUATION CATALYST ANALYSIS
What catalyst could cause multiple expansion from current levels?
What catalyst could cause multiple compression?
What is the most likely multiple range in 12 months under base case?

VALUATION SCORE: X/5

INVESTMENT IMPLICATION: [2-3 sentences]

{_BRIEFING_FOOTER}"""


def p4_sotp_valuation(ticker: str, company_name: str, phase1_3_briefings: str, raw_data: dict) -> str:
    price = raw_data.get('market', {}).get('price', {}).get('current', 'N/A')
    market_cap = raw_data.get('market', {}).get('valuation', {}).get('market_cap', 'N/A')
    return f"""You are a senior equity research analyst specializing in sum-of-the-parts (SOTP) valuation of diversified businesses.

TASK: Sum-of-the-parts valuation for {company_name} ({ticker}).
Current Price: ${price} | Market Cap: ${market_cap:,}

PHASE 1-3 BRIEFINGS:
{phase1_3_briefings}

SECTION 1 -- BUSINESS SEGMENT IDENTIFICATION
List all identifiable business segments/revenue streams with:
- Revenue ($M), estimated Operating Income ($M), estimated EBITDA ($M)
- Growth rate (last year and 3-year CAGR)
- Key pure-play comparables for valuation

SECTION 2 -- SEGMENT VALUATION (Bear/Base/Bull for each)
For each segment, select the most appropriate valuation methodology:
- High-growth / SaaS / subscription: EV/Revenue or EV/ARR
- Profitable software / media: EV/EBITDA or P/E
- Capital-intensive / industrial: EV/EBITDA or P/Assets
- E-commerce / marketplace: EV/GMV or EV/Revenue

| Segment | Metric ($M) | Bear Multiple | Bear Value | Base Multiple | Base Value | Bull Multiple | Bull Value |

Justify your multiple for each segment vs. pure-play peer.

SECTION 3 -- ADJUSTMENTS TO SOTP
Add: Net Cash/(Debt) position ($M)
Add: Value of investments, minority interests, real estate (if material)
Add: Estimated tax assets (NOLs, etc.)
Subtract: Corporate overhead not allocated to segments (capitalize at appropriate multiple)
Subtract: Conglomerate discount (typically 10-20% for diversified businesses): X%
Subtract: Capitalized SBC cost (annual SBC x appropriate P/E multiple)

SECTION 4 -- SOTP SUMMARY
| Component | Bear ($M) | Base ($M) | Bull ($M) |
[Each segment + adjustments]
Total Enterprise Value (Bear/Base/Bull)
Less: Net Debt
Total Equity Value (Bear/Base/Bull)
Shares Outstanding: X billion
Per Share Value: Bear $X / Base $X / Bull $X

SECTION 5 -- SOTP DISCOUNT/PREMIUM
Current market cap vs. SOTP base case: trading at X% discount or X% premium.
If discount: which segment is the market undervaluing? Why?
If premium: what growth optionality is the market pricing in beyond current segment values?

VALUATION SCORE: X/5

INVESTMENT IMPLICATION: [2-3 sentences]

{_BRIEFING_FOOTER}"""


def p4_revenue_evolution_model(ticker: str, company_name: str, phase1_3_briefings: str, raw_data: dict) -> str:
    price = raw_data.get('market', {}).get('price', {}).get('current', 'N/A')
    return f"""You are a senior equity research analyst specializing in revenue modeling, stream analysis, and margin evolution.

TASK: Revenue evolution model for {company_name} ({ticker}).
Current Price: ${price}

PHASE 1-3 BRIEFINGS:
{phase1_3_briefings}

SECTION 1 -- REVENUE STREAM DEEP DIVE
For each revenue stream identified in prior phases:
Revenue quality ranking (1-10): based on predictability, margins, growth durability.
For the top 3 streams, model:
- Current revenue run rate ($M annualized)
- Pricing power assessment (can they raise prices? By how much per year?)
- Volume growth drivers (new customers, expansion, share gain)
- Churn/retention rate (if applicable)
- Market saturation risk

SECTION 2 -- 5-YEAR REVENUE PROJECTION BY STREAM
Base case annual projection for each stream:
| Stream | Year 1 | Year 2 | Year 3 | Year 4 | Year 5 | CAGR | Justification |
Show revenue mix shift over 5 years.
Total revenue: Year 1-5, with compounded growth rate.
Bull case (top-line assumptions): Year 5 revenue target.
Bear case (downside scenario): Year 5 revenue target.

SECTION 3 -- MARGIN EVOLUTION MODEL
Start with current gross/operating margins by segment.
Project blended margins as revenue mix shifts:
| Year | Revenue Mix (% high-margin) | Blended Gross Margin | Blended Op Margin | Est. FCF Margin |
Key margin drivers: scale economies, pricing, product mix, opex leverage.
At Year 5 base case, what are the blended operating and FCF margins?

SECTION 4 -- REVENUE QUALITY RANKING
For each stream, rank on a 1-10 scale for investment quality:
- Predictability (10 = fully recurring/contracted)
- Margin (10 = highest margin stream)
- Growth durability (10 = 10+ year secular tailwind)
- Moat (10 = extremely hard to replicate)
- AI disruption resilience (10 = AI strengthens this stream)

SECTION 5 -- STREAM-WEIGHTED SOTP VALUATION
Using Year 3 forward revenue for each stream:
Apply quality-adjusted multiples (higher quality streams get premium multiples).
| Stream | Year 3 Revenue | Quality Score | Assigned Multiple | Stream Value ($M) |
Total enterprise value -> per-share value.
Bull/Base/Bear per-share targets.

VALUATION SCORE: X/5

INVESTMENT IMPLICATION: [2-3 sentences]

{_BRIEFING_FOOTER}"""


# ===========================================================================
# PHASE 5: FORWARD INTELLIGENCE (Agents 19-22)
# ===========================================================================

def p5_sector_vulnerability_mapper(ticker: str, company_name: str, phase1_4_briefings: str, raw_data: dict) -> str:
    return f"""You are a senior equity research analyst specializing in sector analysis and competitive vulnerability mapping.

TASK: Sector vulnerability mapping for {company_name} ({ticker}).

PHASE 1-4 BRIEFINGS:
{phase1_4_briefings}

SECTION 1 -- SECTOR EXPOSURE MAP
Identify all sectors/industries where {company_name} operates.
For each sector:
| Sector | Revenue Exposure % | Competitive Position (1-5) | Market Share | Share Trend |

SECTION 2 -- VULNERABILITY ASSESSMENT BY SECTOR
For each sector, score vulnerability 1-5 (5 = highest vulnerability):
| Sector | Competition Vulnerability | Regulatory Vulnerability | Macro Sensitivity | AI Impact | Composite Score |
Competition Vulnerability: are new entrants or incumbent attackers gaining share?
Regulatory Vulnerability: pending legislation, existing regulatory risk, compliance burden.
Macro Sensitivity: cyclicality, interest rate sensitivity, consumer spending sensitivity.
AI Impact: is AI a net threat or opportunity in this sector?

SECTION 3 -- REVENUE-WEIGHTED OVERALL VULNERABILITY
Weight each sector's vulnerability score by revenue exposure.
Overall Composite Vulnerability Score: X/5.
Most vulnerable segment: [name, score, primary concern].
Most resilient segment: [name, score, primary strength].

SECTION 4 -- PEER RESILIENCE RANKING
For 3-5 peers in {company_name}'s primary sector:
| Company | Revenue at Risk ($B) | Composite Resilience Score | Key Differentiator |
Where does {ticker} rank on composite resilience vs. peers?

SECTION 5 -- SECTOR vs COMPANY-SPECIFIC RETURN DECOMPOSITION
Estimate: what % of {ticker}'s historical returns were due to sector tailwinds vs. company outperformance?
If sector tailwinds reverse, how would {ticker} perform on company-specific merits alone?
Key sector-level risk over next 12-18 months.

SECTION 6 -- MONITORING FRAMEWORK
For each major vulnerability: the specific metric or event that would signal it's materializing.
Dashboard: what are the top 5 sector KPIs to monitor quarterly?

VULNERABILITY SCORE: X/5 (1=highly resilient, 5=highly vulnerable)

INVESTMENT IMPLICATION: [2-3 sentences]

{_BRIEFING_FOOTER}"""


def p5_competitive_threat_tracker(ticker: str, company_name: str, phase1_4_briefings: str, raw_data: dict) -> str:
    return f"""You are a senior equity research analyst specializing in competitive intelligence and market share dynamics.

TASK: Competitive threat analysis for {company_name} ({ticker}).

PHASE 1-4 BRIEFINGS:
{phase1_4_briefings}

SECTION 1 -- COMPETITIVE THREAT MATRIX
Identify 8-10 specific competitive threats (by company or technology category).
| # | Threat Source | Probability (1-5) | Severity (1-5) | Threat Score | Type | Timeline |
Type: Share Taker (taking existing revenue), TAM Expander (growing the pie), Disruptive Substitute (replacing the category).
Threat Score = Probability x Severity (max 25).
Sort by Threat Score descending.

SECTION 2 -- DEEP DIVE: TOP 3 COMPETITIVE THREATS
For each of the three highest-scoring threats:
Company/Technology: [Name]
Current Status: [Revenue, funding, growth rate, user base -- be specific with numbers]
Mechanism: [How exactly does this threaten {ticker}'s business?]
Revenue at Risk: [Estimate $M of {ticker} revenue at risk in 3 years]
Management Response: [What is {ticker} doing to counter this threat?]
Timeline: [When does this threat become material?]
Monitoring Signal: [What would you look for in quarterly results?]

SECTION 3 -- COMPETITIVE POSITIONING MAP
Plot {ticker} vs. top 5 competitors on two dimensions:
Dimension 1: Product breadth/integration (narrow to broad)
Dimension 2: Price competitiveness (premium to value)
Is {ticker}'s position in the favorable or unfavorable quadrant?

SECTION 4 -- AI DISRUPTION ASSESSMENT BY BUSINESS LINE
For each major business line:
| Business Line | Revenue ($M) | Disruptor | AI Mechanism | Probability | Timeline | Net Impact |
Classify each as: Net Opportunity (AI makes line stronger), Mixed, or Net Threat.
Revenue-weighted overall AI disruption score.

SECTION 5 -- COMPETITIVE DYNAMICS TRENDS
Is the competitive landscape becoming more or less favorable?
New entrants vs. consolidation trend.
Platform vs. point solution competitive dynamics (who wins?).
International competitive dynamics.

OVERALL COMPETITIVE THREAT SCORE: X/5 (1=dominant competitive position, 5=under severe competitive pressure)

KEY FINDING: What is the single most important competitive threat, and why? [2-3 sentences]

{_BRIEFING_FOOTER}"""


def p5_scenario_progression_mapper(ticker: str, company_name: str, phase1_4_briefings: str, raw_data: dict) -> str:
    price = raw_data.get('market', {}).get('price', {}).get('current', 'N/A')
    return f"""You are a senior equity research analyst specializing in scenario analysis, forward modeling, and catalyst tracking.

TASK: Scenario progression mapping for {company_name} ({ticker}).
Current Price: ${price}

PHASE 1-4 BRIEFINGS:
{phase1_4_briefings}

SECTION 1 -- THE ONE BRANCHING METRIC
Identify the single metric that will most determine whether the bull or bear case plays out.
Why this metric? What is its current value? What would bull/base/bear levels look like?
This is the North Star metric for this investment thesis.

SECTION 2 -- THREE SCENARIOS WITH PROBABILITY WEIGHTS
BULL CASE (probability: X%):
Narrative: [2-3 sentences on what goes right]
Price Target: $X (12-month), $X (24-month)
Required conditions: [3-4 specific measurable things that must happen]
Confidence level: Low / Medium / High

BASE CASE (probability: X%): [same structure]

BEAR CASE (probability: X%): [same structure]

Note: Probabilities must sum to 100%.

SECTION 3 -- QUARTERLY PROGRESSION MAP (next 6-8 quarters)
For each scenario, map out the expected trajectory quarter by quarter:
| Quarter | Bull: Key Metrics | Base: Key Metrics | Bear: Key Metrics | Branching Point |
Key metrics: Revenue growth YoY, Operating Margin, EPS, FCF Margin, Stock Price range.
Branching points: specific quarters where trajectories diverge most sharply.

SECTION 4 -- CATALYST CALENDAR (next 18 months)
| Date / Quarter | Event | Expected Direction | Magnitude (High/Med/Low) | Scenario Impact |
Events: earnings dates, product launches, analyst days, regulatory decisions, macro events,
contract renewals, competitor product launches, key executive departures/hires.
For each catalyst: if it plays out as expected, which scenario does it confirm?

SECTION 5 -- WHAT WOULD ACCELERATE THE BULL CASE?
3 specific events that would pull forward the bull case timeline.
For each: probability, timeframe, price impact estimate.

FORWARD CONVICTION SCORE: X/5 (5 = high conviction bull/bear, 1 = highly uncertain)

INVESTMENT IMPLICATION: [2-3 sentences]

{_BRIEFING_FOOTER}"""


def p5_thesis_narrative_builder(ticker: str, company_name: str, phase1_4_briefings: str, raw_data: dict) -> str:
    price = raw_data.get('market', {}).get('price', {}).get('current', 'N/A')
    return f"""You are a senior portfolio manager writing an investment thesis memo for an institutional audience.

TASK: Write the investment thesis narrative for {company_name} ({ticker}).
Current Price: ${price}

PHASE 1-4 BRIEFINGS:
{phase1_4_briefings}

Write this as a polished, institutional-quality investment memo. Be direct, specific, and avoid hedging
without substance. Every claim must be backed by a specific number or evidence from the briefings.

SECTION 1 -- THE CRITICAL QUESTION
Open with the single most important question the market is currently debating about this stock.
This should be the question that, when answered, resolves the bull vs. bear debate.
Format: "The critical question for [company]: [Question]"

SECTION 2 -- OUR POSITION
State which side of the debate we're on and our conviction level.
One clear declarative sentence: "We are [bullish/bearish/neutral on {ticker}] because [core reason]."

SECTION 3 -- THE FULL THESIS
Write the investment thesis as a flowing narrative (400-600 words):
a) SETUP: What is the market consensus view, and why is the market currently
   [over/under]valuing this company?
b) OUR VIEW: What do we see that the market is missing or overweighting?
c) THE EVIDENCE: The 3-4 most compelling pieces of evidence for our position.
   Use specific numbers. Be precise.
d) THE TRADE: How does the opportunity play out? What is the path to the stock
   reflecting our view? What is the expected return and time horizon?

SECTION 4 -- WHAT WOULD CHANGE OUR MIND
3-4 specific, measurable conditions that would cause us to reverse our position.
Format: "We would change our view to [bearish/bullish] if: [specific condition with numbers]"
These should be genuine threshold conditions, not vague concerns.

SECTION 5 -- ACTION FRAMEWORK
Entry Level: $X -- [why this is the right entry level]
Initial Size: X% of portfolio -- [position sizing rationale]
Add Level: $X -- [what would cause us to add more?]
Full Size: X% -- [what would justify maximum conviction?]
Stop/Cut Level: $X -- [what price/condition signals we were wrong?]
Time Horizon: X months -- [catalysts or milestones that define the horizon]
Base Case Target: $X (+X% from current)
Bull Case Target: $X (+X% from current)
Bear Case Target: $X (-X% from current)

{_BRIEFING_FOOTER}"""


# ===========================================================================
# PHASE 6: SYNTHESIS (Agents 23-25)
# ===========================================================================

def p6_bull_case_builder(ticker: str, company_name: str, all_prior_briefings: str, phase5_full: str) -> str:
    return f"""You are a senior equity research analyst building the bull case for {company_name} ({ticker}).

ALL PRIOR PHASE BRIEFINGS:
{all_prior_briefings}

PHASE 5 DETAILED OUTPUT (for reference):
{phase5_full[:3000] if len(phase5_full) > 3000 else phase5_full}

TASK: Build the most rigorous, evidence-based bull case possible for {ticker}.
You should steel-man the bull argument -- present it at its strongest, not a strawman.
Every argument must be backed by specific numbers from the research above.

SECTION 1 -- THE FIVE STRONGEST BULL ARGUMENTS
For each of the 5 bull arguments (ranked by probability x impact):

ARGUMENT #X: [Name of argument in bold]
Evidence: [Specific numbers, data points, and facts that support this argument]
"If This Plays Out" Scenario:
  - Impact on revenue/earnings ($ and %)
  - Impact on stock price ($ and %)
  - Timeline for realization
Confirmation Trigger: [The specific metric or event that confirms this argument is playing out]
Invalidation Trigger: [The specific metric or event that proves this argument wrong]
Quarterly Progression: [If this argument materializes, what do Q1-Q4 results look like?]

SECTION 2 -- BULL CASE AGGREGATE ASSESSMENT
Aggregate probability that the bull case plays out: X%
Bull Case 12-month price target: $X
Bull Case 24-month price target: $X
Expected return to 12-month bull target: +X%

SECTION 3 -- BULL CASE DEPENDENCIES
List the 3-4 things that MUST go right for the full bull case to play out.
Are these dependent on each other, or independent?
What is the probability that ALL required conditions are met simultaneously?

SECTION 4 -- BULL CASE RISK-REWARD
If bull case: +X% return. If base case: +X% return. If bear case: -X% return.
With probability weights: Expected Value = +X% from current price.

SECTION 5 -- WHAT THE MARKET IS MISSING
The single most important thing the market is currently underweighting in the bull case.
Why does this mispricing exist? Who is on the other side of this trade?

BULL CASE AGGREGATE PROBABILITY: X%

{_BRIEFING_FOOTER}"""


def p6_bear_case_builder(ticker: str, company_name: str, all_prior_briefings: str, phase5_full: str) -> str:
    return f"""You are a senior equity research analyst building the bear case for {company_name} ({ticker}).

ALL PRIOR PHASE BRIEFINGS:
{all_prior_briefings}

PHASE 5 DETAILED OUTPUT (for reference):
{phase5_full[:3000] if len(phase5_full) > 3000 else phase5_full}

TASK: Build the most rigorous, evidence-based bear case for {ticker}.
Steel-man the bear argument. Present it at maximum strength with specific evidence.
Do not be dismissive of real risks -- institutional short sellers have done this work.

SECTION 1 -- THE FIVE STRONGEST BEAR ARGUMENTS
For each of the 5 bear arguments (ranked by probability x impact):

ARGUMENT #X: [Name of argument in bold]
Evidence: [Specific numbers, data points, facts supporting this risk]
"If This Plays Out" Scenario:
  - Impact on revenue/earnings ($ and %)
  - Downside to stock price ($ and %)
  - Timeline for realization
Confirmation Trigger: [Specific metric or event that confirms this risk is materializing]
Invalidation Trigger: [Specific metric or event that proves this risk is overblown]
Quarterly Progression: [If this argument materializes, what do Q1-Q4 results look like?]

SECTION 2 -- BEAR CASE AGGREGATE ASSESSMENT
Aggregate probability that the bear case plays out: X%
Bear Case 12-month price target: $X
Bear Case 24-month price target: $X
Expected loss to 12-month bear target: -X%

SECTION 3 -- CATALYST CALENDAR FOR BEAR CASE
| Date/Quarter | Catalyst | Bear Case Trigger | Expected Price Impact |
List 6-8 specific dates/events in the next 18 months that could trigger the bear case.

SECTION 4 -- MARKET MISPRICING ANALYSIS
Which bear argument does THE MARKET currently overweight? (i.e., which fear is already priced in?)
Which bear argument does OUR ANALYSIS take most seriously? (i.e., what is underpriced risk?)
If the market-overweighted fear doesn't materialize, what is the upside?

SECTION 5 -- SHORT SELLER FRAMEWORK
If forced to build a short thesis, what are the 3 key arguments?
What is the expected timeline for the short thesis to play out?
What event would cause a short squeeze (rapid covering)?

BEAR CASE AGGREGATE PROBABILITY: X%

{_BRIEFING_FOOTER}"""


def p6_final_synthesis(ticker: str, company_name: str, all_prior_briefings: str, valuation_details: str) -> str:
    return f"""You are the Chief Investment Officer producing the final investment verdict for {company_name} ({ticker}).

ALL PRIOR PHASE BRIEFINGS (Phases 1-6):
{all_prior_briefings}

VALUATION DETAILS (for convergence analysis):
{valuation_details[:2000] if len(valuation_details) > 2000 else valuation_details}

TASK: Produce the final synthesis report -- the ultimate investment verdict.
This is the definitive document. It must be comprehensive, precise, and actionable.

SECTION 1 -- DIMENSIONAL SCORECARD
Compile all scores from prior agents into a single table:
| Dimension | Agent | Score (1-5) | Weight | Weighted Score |
Phase 1: SEC Data Quality, Earnings Quality, Market Structure, News Sentiment
Phase 2: Income Quality, Balance Sheet, Cash Flow, Segment Analysis, Historical Trends
Phase 3: Management, Moat, Industry, Risk
Phase 4: DCF Attractiveness, Comps, Historical Multiples, SOTP, Revenue Evolution
Phase 5: Vulnerability, Competitive Threat, Forward Conviction
Phase 6: Bull Probability, Bear Probability

COMPOSITE SCORE: X.X / 5.0

SECTION 2 -- VALUATION CONVERGENCE TABLE
| Method | Weight | Bear Value | Base Value | Bull Value |
DCF (40%), Comparable Companies (30%), Historical Multiples (10%), Revenue Evolution (20%)
Blended Fair Value: Bear $X / Base $X / Bull $X
Current Price: $X
Discount/(Premium) to base case: X%

SECTION 3 -- PROBABILITY-WEIGHTED EXPECTED VALUE
| Scenario | Probability | Price Target | Expected Value Contribution |
Bull case: X% x $X = $X contribution
Base case: X% x $X = $X contribution
Bear case: X% x $X = $X contribution
Probability-weighted expected value: $X
Expected return from current price: +X% / -X%

SECTION 4 -- RISK/REWARD RATIO
Upside to Base Case: +X%
Downside to Bear Case: -X%
Risk/Reward Ratio: X:1 (upside to base / downside to bear)
For every dollar of downside risk, there is $X of base case upside.

SECTION 5 -- FINAL RATING
RATING: [STRONG BUY / BUY / HOLD / SELL / STRONG SELL]
CONVICTION: [Low / Medium / High]
Justification: [3-4 sentences explaining the rating, addressing the most important factors]

SECTION 6 -- PORTFOLIO DECISION FRAMEWORK
Entry Level: $X -- Why: [rationale]
Initial Position Size: X% -- Why: [rationale]
Add Level: $X if [specific condition]
Full Position Size: X% -- Condition: [what must be true to go full size]
Stop/Cut Level: $X -- Rationale: [why this level invalidates the thesis]
Time Horizon: X months

SECTION 7 -- UPGRADE/DOWNGRADE TRIGGERS
UPGRADE to [higher rating] if:
  1. [Specific measurable condition #1]
  2. [Specific measurable condition #2]
  3. [Specific measurable condition #3]

DOWNGRADE to [lower rating] if:
  1. [Specific measurable condition #1]
  2. [Specific measurable condition #2]
  3. [Specific measurable condition #3]

SECTION 8 -- INVESTMENT HORIZON OUTLOOK
Short Term (0-6 months): [catalysts, risks, expected price range]
Medium Term (6-18 months): [thesis progression, key milestones]
Long Term (18+ months): [secular story, terminal value drivers]

{_BRIEFING_FOOTER}"""


# ===========================================================================
# PHASE 7: ACADEMIC EVIDENCE RATING (Agent 26)
# ===========================================================================

def p7_evidence_based_rating(ticker: str, company_name: str, all_briefings: str, phase6_synthesis: str) -> str:
    return f"""You are an academic finance researcher with deep expertise in quantitative factor models and empirical asset pricing.

TASK: Produce an evidence-based investment rating for {company_name} ({ticker}) based EXCLUSIVELY on
factors that academic finance literature has shown to be statistically predictive of future stock returns.

ALL RESEARCH BRIEFINGS (Phases 1-6):
{all_briefings}

PHASE 6 FINAL SYNTHESIS (for reference):
{phase6_synthesis[:3000] if len(phase6_synthesis) > 3000 else phase6_synthesis}

METHODOLOGY:
For each factor below, assign a score of -2 to +2 based ONLY on evidence in the research above:
  -2 = Strongly bearish signal (strong academic evidence points to underperformance)
  -1 = Mildly bearish signal
   0 = Neutral (factor is inconclusive or data unavailable)
  +1 = Mildly bullish signal
  +2 = Strongly bullish signal (strong academic evidence points to outperformance)
Note: If a specific data point is not available, score 0 and explain.

FACTOR 1 -- VALUE FACTOR (Weight: 20%) [Fama & French, 1992; Fama & French, 1993]
Academic basis: Stocks with low P/B, low P/E, and high book-to-market ratios systematically
outperform growth stocks over long periods.
Evidence from research: [P/E vs history and peers, P/B, EV/EBITDA vs peers]
Score: X (-2 to +2) | Weighted Contribution: X x 0.20 = X
Rationale: [2-3 sentences citing specific valuation metrics from the research]

FACTOR 2 -- MOMENTUM FACTOR (Weight: 10%) [Jegadeesh & Titman, 1993]
Academic basis: Stocks with positive 6-12 month price momentum tend to continue outperforming
over the next 3-12 months. Avoid stocks with poor 1-month momentum (reversal).
Evidence from research: [6-month and 12-month price performance, trend direction]
Score: X | Weighted Contribution: X x 0.10 = X
Rationale: [2-3 sentences on price momentum]

FACTOR 3 -- QUALITY FACTOR (Weight: 20%) [Novy-Marx, 2013; Asness, Frazzini & Pedersen, 2013]
Academic basis: High-quality companies (high profitability, low leverage, stable earnings)
outperform low-quality companies even when expensive on traditional metrics.
Quality signals: Gross Profit / Assets, ROE stability, earnings predictability, low leverage.
Evidence from research: [Gross margins, ROE, debt levels, earnings volatility]
Score: X | Weighted Contribution: X x 0.20 = X
Rationale: [2-3 sentences on quality signals]

FACTOR 4 -- EARNINGS QUALITY / ACCRUALS ANOMALY (Weight: 10%) [Sloan, 1996]
Academic basis: Companies with high accruals (net income significantly exceeds OCF) tend to
underperform. Low accruals (OCF >> net income) predict outperformance.
Sloan Accruals Ratio = (Net Income - OCF) / Average Total Assets. Below -0.05 = bullish.
Evidence from research: [OCF/Net Income ratio, accruals ratio from Phase 2]
Score: X | Weighted Contribution: X x 0.10 = X
Rationale: [2-3 sentences on accruals and earnings quality]

FACTOR 5 -- ROIC vs WACC SPREAD (Weight: 15%) [Greenwald, Judd & Kahn; EVA literature]
Academic basis: Companies that consistently earn ROIC above WACC create real economic value
and tend to generate superior long-term returns. The spread predicts sustained outperformance.
Evidence from research: [ROIC estimates, WACC estimate, spread, trend]
Score: X | Weighted Contribution: X x 0.15 = X
Rationale: [2-3 sentences on economic value creation]

FACTOR 6 -- INSIDER BEHAVIOR (Weight: 5%) [Lakonishok & Lee, 2001; Seyhun, 1988]
Academic basis: Net insider buying (especially by CEO/CFO) is statistically predictive of
positive future returns. Open-market purchases (not option exercises) have the strongest signal.
Evidence from research: [insider transaction data from Phase 1]
Score: X | Weighted Contribution: X x 0.05 = X
Rationale: [2-3 sentences on insider activity]

FACTOR 7 -- ANALYST REVISION DIRECTION (Weight: 5%) [Chan, Jegadeesh & Lakonishok, 1996]
Academic basis: Upward earnings estimate revisions are predictive of future outperformance,
particularly in the 3-12 months following the revision. Downward revisions predict underperformance.
Evidence from research: [estimate revision trend from Phase 1 market data agent]
Score: X | Weighted Contribution: X x 0.05 = X
Rationale: [2-3 sentences on revision trend]

FACTOR 8 -- COMPETITIVE MOAT DURABILITY (Weight: 15%) [Greenwald & Kahn, 2005]
Academic basis: Companies with wide, widening competitive moats generate sustained above-market
returns because they can earn above-WACC returns for extended periods.
Evidence from research: [moat score, moat trend from Phase 3]
Score: X | Weighted Contribution: X x 0.15 = X
Rationale: [2-3 sentences on moat quality and durability]

WEIGHTED COMPOSITE SCORE:
| Factor | Raw Score | Weight | Contribution |
[table for all 8 factors]
Total Weighted Score: X.XX

RATING MAPPING:
  Score > +0.80: BUY
  Score +0.30 to +0.80: LEAN BUY (moderate conviction)
  Score -0.30 to +0.30: HOLD
  Score -0.80 to -0.30: LEAN SELL
  Score < -0.80: SELL

ACADEMIC EVIDENCE RATING: [BUY / LEAN BUY / HOLD / LEAN SELL / SELL]

CONFIDENCE LEVEL: [Low / Medium / High]
Rationale: [Confidence is High if 5+ factors point in the same direction,
Medium if 3-4 factors agree, Low if factors are mixed or data is sparse]

THREE MOST DECISIVE FACTORS:
1. [Factor name]: [Why this was most important for the rating]
2. [Factor name]: [Why this was important]
3. [Factor name]: [Why this was important]

CONFLICTS WITH NARRATIVE ANALYSIS:
If any academic factor signal contradicts the narrative thesis from prior phases, note it explicitly.
Format: "The [factor] factor gives a [bearish/bullish] signal of [X], which contradicts the
[narrative position] from the synthesis because [explanation]."

ACADEMIC EVIDENCE RATING: [BUY/HOLD/SELL] with [X]% confidence

{_BRIEFING_FOOTER}"""
