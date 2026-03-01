from __future__ import annotations

"""
Data collection module for the Deep Equity Research System.

Four collector classes:
  - SECCollector      : SEC EDGAR filings and XBRL financial data
  - MarketDataCollector : yfinance market/fundamental data
  - NewsCollector     : DuckDuckGo news search + BeautifulSoup parsing
  - EarningsCollector : Earnings dates, estimates, and 8-K links
"""

import json
import logging
import time
import re
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import quote_plus

import requests
import yfinance as yf
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

from config import (
    SEC_HEADERS,
    SEC_BASE_URL,
    SEC_TICKERS_URL,
    DDG_MAX_RESULTS,
    REQUEST_TIMEOUT,
    YFINANCE_PERIOD_1Y,
    YFINANCE_PERIOD_5Y,
    YFINANCE_INTERVAL_DAILY,
    YFINANCE_INTERVAL_MONTHLY,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _safe_get(d: dict, *keys, default=None):
    """Safe nested dict access."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
        if d is None:
            return default
    return d


def _http_get(url: str, headers: dict, params: dict | None = None, timeout: int = REQUEST_TIMEOUT) -> requests.Response:
    """Make an HTTP GET request with error logging."""
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp
    except requests.RequestException as e:
        logger.error("HTTP GET failed for %s: %s", url, str(e)[:300])
        raise


# ---------------------------------------------------------------------------
# SEC EDGAR Collector
# ---------------------------------------------------------------------------

class SECCollector:
    """
    Fetches filings metadata and XBRL financial data from SEC EDGAR.

    Flow:
      1. Load company_tickers.json to map ticker → CIK
      2. Fetch submissions JSON for recent 10-K / 10-Q filing URLs
      3. Fetch XBRL companyfacts JSON for structured financials
    """

    _tickers_cache: dict | None = None  # class-level cache

    def __init__(self):
        self.base_url = SEC_BASE_URL
        self.headers = SEC_HEADERS

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def get_data(self, ticker: str) -> dict:
        """Return a comprehensive dict of SEC data for the given ticker."""
        ticker = ticker.upper().strip()
        result: dict[str, Any] = {
            "ticker": ticker,
            "cik": None,
            "company_name": None,
            "recent_10k_urls": [],
            "recent_10q_urls": [],
            "financials": {},
            "errors": [],
        }

        try:
            cik, company_name = self._lookup_cik(ticker)
        except Exception as e:
            result["errors"].append(f"CIK lookup failed: {e}")
            logger.warning("Could not find CIK for %s: %s", ticker, e)
            return result

        result["cik"] = cik
        result["company_name"] = company_name
        padded_cik = str(cik).zfill(10)

        # --- Submissions (filing index) ---
        try:
            submissions = self._fetch_submissions(padded_cik)
            result["recent_10k_urls"] = self._extract_filing_urls(submissions, "10-K", 3)
            result["recent_10q_urls"] = self._extract_filing_urls(submissions, "10-Q", 4)
        except Exception as e:
            result["errors"].append(f"Submissions fetch failed: {e}")
            logger.warning("Submissions error for %s: %s", ticker, e)

        # --- XBRL Company Facts ---
        try:
            facts = self._fetch_company_facts(padded_cik)
            result["financials"] = self._parse_xbrl_facts(facts)
        except Exception as e:
            result["errors"].append(f"XBRL facts fetch failed: {e}")
            logger.warning("XBRL error for %s: %s", ticker, e)

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _lookup_cik(self, ticker: str) -> tuple[int, str]:
        """Map ticker to (cik, company_name) using SEC's company_tickers.json."""
        if SECCollector._tickers_cache is None:
            resp = _http_get(SEC_TICKERS_URL, self.headers)
            SECCollector._tickers_cache = resp.json()

        data = SECCollector._tickers_cache
        for _idx, entry in data.items():
            if entry.get("ticker", "").upper() == ticker:
                return int(entry["cik_str"]), entry.get("title", ticker)

        raise ValueError(f"Ticker '{ticker}' not found in SEC company_tickers.json")

    def _fetch_submissions(self, padded_cik: str) -> dict:
        url = f"{self.base_url}/submissions/CIK{padded_cik}.json"
        resp = _http_get(url, self.headers)
        return resp.json()

    def _fetch_company_facts(self, padded_cik: str) -> dict:
        url = f"{self.base_url}/api/xbrl/companyfacts/CIK{padded_cik}.json"
        resp = _http_get(url, self.headers)
        return resp.json()

    def _extract_filing_urls(self, submissions: dict, form_type: str, limit: int) -> list[str]:
        """Extract the most recent filing index URLs for the given form type."""
        filings = _safe_get(submissions, "filings", "recent", default={})
        forms = filings.get("form", [])
        accessions = filings.get("accessionNumber", [])
        cik = submissions.get("cik", "")

        urls = []
        for form, acc in zip(forms, accessions):
            if form == form_type:
                acc_clean = acc.replace("-", "")
                url = (
                    f"https://www.sec.gov/Archives/edgar/data/{cik}/"
                    f"{acc_clean}/{acc}-index.htm"
                )
                urls.append(url)
                if len(urls) >= limit:
                    break
        return urls

    def _parse_xbrl_facts(self, facts: dict) -> dict:
        """
        Extract key financial metrics from XBRL company facts.
        Looks for US-GAAP and DEI concepts.
        """
        us_gaap = _safe_get(facts, "facts", "us-gaap", default={})
        dei = _safe_get(facts, "facts", "dei", default={})

        def _latest_annual_values(concept_data: dict, n: int = 4) -> list[dict]:
            """Get the n most recent annual (10-K) values for a concept."""
            units = concept_data.get("units", {})
            # Try USD first, then shares, then USD/shares, then pure
            for unit_key in ("USD", "shares", "USD/shares", "pure"):
                if unit_key in units:
                    entries = units[unit_key]
                    # Get all 10-K entries (frame field is optional)
                    annual = [
                        e for e in entries
                        if e.get("form") in ("10-K", "10-K/A")
                    ]
                    # Sort by end date descending
                    annual.sort(key=lambda x: x.get("end", ""), reverse=True)
                    # De-duplicate by end date (keep first = most recent filing)
                    seen = {}
                    for e in annual:
                        end = e.get("end", "")
                        if end and end not in seen:
                            seen[end] = e
                    deduped = list(seen.values())[:n]
                    return [{"end": e.get("end"), "val": e.get("val"), "unit": unit_key} for e in deduped]
            return []

        concepts = {
            # Income Statement
            "Revenues": "revenue",
            "RevenueFromContractWithCustomerExcludingAssessedTax": "revenue_alt",
            "SalesRevenueNet": "revenue_net",
            "CostOfRevenue": "cost_of_revenue",
            "GrossProfit": "gross_profit",
            "OperatingExpenses": "operating_expenses",
            "OperatingIncomeLoss": "operating_income",
            "NetIncomeLoss": "net_income",
            "EarningsPerShareBasic": "eps_basic",
            "EarningsPerShareDiluted": "eps_diluted",
            "ResearchAndDevelopmentExpense": "rd_expense",
            "SellingGeneralAndAdministrativeExpense": "sga_expense",
            "ShareBasedCompensation": "stock_based_compensation",
            "DepreciationAndAmortization": "depreciation_amortization",
            # Balance Sheet
            "CashAndCashEquivalentsAtCarryingValue": "cash",
            "ShortTermInvestments": "short_term_investments",
            "AccountsReceivableNetCurrent": "accounts_receivable",
            "InventoryNet": "inventory",
            "AssetsCurrent": "current_assets",
            "PropertyPlantAndEquipmentNet": "ppe_net",
            "Goodwill": "goodwill",
            "IntangibleAssetsNetExcludingGoodwill": "intangibles",
            "Assets": "total_assets",
            "LiabilitiesCurrent": "current_liabilities",
            "LongTermDebt": "long_term_debt",
            "LongTermDebtNoncurrent": "long_term_debt_noncurrent",
            "Liabilities": "total_liabilities",
            "StockholdersEquity": "stockholders_equity",
            # Cash Flow
            "NetCashProvidedByUsedInOperatingActivities": "operating_cash_flow",
            "PaymentsToAcquirePropertyPlantAndEquipment": "capex",
            "NetCashProvidedByUsedInInvestingActivities": "investing_cash_flow",
            "NetCashProvidedByUsedInFinancingActivities": "financing_cash_flow",
            # Share counts
            "CommonStockSharesOutstanding": "shares_outstanding",
            "WeightedAverageNumberOfSharesOutstandingBasic": "shares_basic",
            "WeightedAverageNumberOfDilutedSharesOutstanding": "shares_diluted",
        }

        parsed: dict[str, Any] = {}
        for gaap_concept, field_name in concepts.items():
            if gaap_concept in us_gaap:
                parsed[field_name] = _latest_annual_values(us_gaap[gaap_concept])

        # DEI concepts
        dei_concepts = {
            "EntityCommonStockSharesOutstanding": "shares_outstanding_dei",
            "EntityPublicFloat": "public_float",
        }
        for dei_concept, field_name in dei_concepts.items():
            if dei_concept in dei:
                parsed[field_name] = _latest_annual_values(dei[dei_concept])

        return parsed


# ---------------------------------------------------------------------------
# Market Data Collector
# ---------------------------------------------------------------------------

class MarketDataCollector:
    """
    Fetches market and fundamental data via yfinance.
    Returns a structured dict covering price, valuation, financials,
    ownership, insider activity, and options.
    """

    def get_data(self, ticker: str) -> dict:
        ticker = ticker.upper().strip()
        result: dict[str, Any] = {
            "ticker": ticker,
            "errors": [],
        }

        try:
            t = yf.Ticker(ticker)
            info = t.info or {}
        except Exception as e:
            result["errors"].append(f"yfinance Ticker init failed: {e}")
            return result

        # --- Price & valuation multiples ---
        result["price"] = {
            "current": info.get("currentPrice") or info.get("regularMarketPrice"),
            "previous_close": info.get("previousClose"),
            "open": info.get("open"),
            "day_low": info.get("dayLow"),
            "day_high": info.get("dayHigh"),
            "week_52_low": info.get("fiftyTwoWeekLow"),
            "week_52_high": info.get("fiftyTwoWeekHigh"),
            "50d_avg": info.get("fiftyDayAverage"),
            "200d_avg": info.get("twoHundredDayAverage"),
            "ytd_return": info.get("52WeekChange"),
        }
        result["valuation"] = {
            "market_cap": info.get("marketCap"),
            "enterprise_value": info.get("enterpriseValue"),
            "pe_trailing": info.get("trailingPE"),
            "pe_forward": info.get("forwardPE"),
            "peg_ratio": info.get("pegRatio"),
            "price_to_sales": info.get("priceToSalesTrailing12Months"),
            "price_to_book": info.get("priceToBook"),
            "ev_to_revenue": info.get("enterpriseToRevenue"),
            "ev_to_ebitda": info.get("enterpriseToEbitda"),
            "beta": info.get("beta"),
        }
        result["fundamentals"] = {
            "revenue_ttm": info.get("totalRevenue"),
            "revenue_growth": info.get("revenueGrowth"),
            "gross_margins": info.get("grossMargins"),
            "operating_margins": info.get("operatingMargins"),
            "profit_margins": info.get("profitMargins"),
            "ebitda": info.get("ebitda"),
            "ebitda_margins": info.get("ebitdaMargins"),
            "eps_trailing": info.get("trailingEps"),
            "eps_forward": info.get("forwardEps"),
            "earnings_growth": info.get("earningsGrowth"),
            "book_value": info.get("bookValue"),
            "debt_to_equity": info.get("debtToEquity"),
            "return_on_equity": info.get("returnOnEquity"),
            "return_on_assets": info.get("returnOnAssets"),
            "free_cashflow": info.get("freeCashflow"),
            "operating_cashflow": info.get("operatingCashflow"),
            "total_debt": info.get("totalDebt"),
            "total_cash": info.get("totalCash"),
            "total_cash_per_share": info.get("totalCashPerShare"),
            "short_ratio": info.get("shortRatio"),
            "short_percent_of_float": info.get("shortPercentOfFloat"),
            "shares_outstanding": info.get("sharesOutstanding"),
            "float_shares": info.get("floatShares"),
            "shares_short": info.get("sharesShort"),
        }
        result["company_info"] = {
            "name": info.get("longName") or info.get("shortName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "description": (info.get("longBusinessSummary") or "")[:1000],
            "employees": info.get("fullTimeEmployees"),
            "country": info.get("country"),
            "website": info.get("website"),
            "exchange": info.get("exchange"),
        }

        # --- Analyst data ---
        try:
            result["analyst"] = {
                "recommendation": info.get("recommendationKey"),
                "recommendation_mean": info.get("recommendationMean"),
                "number_of_analyst_opinions": info.get("numberOfAnalystOpinions"),
                "target_mean_price": info.get("targetMeanPrice"),
                "target_high_price": info.get("targetHighPrice"),
                "target_low_price": info.get("targetLowPrice"),
                "target_median_price": info.get("targetMedianPrice"),
            }
        except Exception as e:
            result["errors"].append(f"Analyst data error: {e}")

        # --- Historical price data ---
        try:
            hist_1y = t.history(period=YFINANCE_PERIOD_1Y, interval=YFINANCE_INTERVAL_DAILY)
            result["price_history_1y"] = self._df_to_records(hist_1y)
        except Exception as e:
            result["errors"].append(f"1y price history error: {e}")

        try:
            hist_5y = t.history(period=YFINANCE_PERIOD_5Y, interval=YFINANCE_INTERVAL_MONTHLY)
            result["price_history_5y"] = self._df_to_records(hist_5y)
        except Exception as e:
            result["errors"].append(f"5y price history error: {e}")

        # --- Financial statements ---
        try:
            result["income_stmt_annual"] = self._df_to_dict(t.income_stmt)
        except Exception as e:
            result["errors"].append(f"Annual income stmt error: {e}")

        try:
            result["income_stmt_quarterly"] = self._df_to_dict(t.quarterly_income_stmt)
        except Exception as e:
            result["errors"].append(f"Quarterly income stmt error: {e}")

        try:
            result["balance_sheet_annual"] = self._df_to_dict(t.balance_sheet)
        except Exception as e:
            result["errors"].append(f"Annual balance sheet error: {e}")

        try:
            result["balance_sheet_quarterly"] = self._df_to_dict(t.quarterly_balance_sheet)
        except Exception as e:
            result["errors"].append(f"Quarterly balance sheet error: {e}")

        try:
            result["cash_flow_annual"] = self._df_to_dict(t.cash_flow)
        except Exception as e:
            result["errors"].append(f"Annual cash flow error: {e}")

        try:
            result["cash_flow_quarterly"] = self._df_to_dict(t.quarterly_cash_flow)
        except Exception as e:
            result["errors"].append(f"Quarterly cash flow error: {e}")

        # --- Analyst recommendations ---
        try:
            recs = t.recommendations
            if recs is not None and not recs.empty:
                result["analyst_recommendations"] = recs.tail(20).to_dict(orient="records")
        except Exception as e:
            result["errors"].append(f"Analyst recommendations error: {e}")

        # --- Institutional holders ---
        try:
            inst = t.institutional_holders
            if inst is not None and not inst.empty:
                result["institutional_holders"] = inst.head(20).to_dict(orient="records")
        except Exception as e:
            result["errors"].append(f"Institutional holders error: {e}")

        # --- Insider transactions ---
        try:
            insider = t.insider_transactions
            if insider is not None and not insider.empty:
                result["insider_transactions"] = insider.head(30).to_dict(orient="records")
        except Exception as e:
            result["errors"].append(f"Insider transactions error: {e}")

        # --- Basic options data (nearest expiry) ---
        try:
            expirations = t.options
            if expirations:
                nearest = expirations[0]
                chain = t.option_chain(nearest)
                result["options"] = {
                    "expiration": nearest,
                    "calls_volume": int(chain.calls["volume"].sum()) if "volume" in chain.calls.columns else None,
                    "puts_volume": int(chain.puts["volume"].sum()) if "volume" in chain.puts.columns else None,
                    "calls_oi": int(chain.calls["openInterest"].sum()) if "openInterest" in chain.calls.columns else None,
                    "puts_oi": int(chain.puts["openInterest"].sum()) if "openInterest" in chain.puts.columns else None,
                }
        except Exception as e:
            result["errors"].append(f"Options data error: {e}")

        return result

    # ------------------------------------------------------------------

    @staticmethod
    def _df_to_dict(df) -> dict:
        """Convert a pandas DataFrame with DatetimeIndex columns to a plain dict."""
        if df is None or df.empty:
            return {}
        try:
            # Convert columns (dates) to strings
            df = df.copy()
            df.columns = [str(c)[:10] for c in df.columns]
            return df.where(df.notna(), None).to_dict()
        except Exception as e:
            logger.warning("DataFrame conversion failed: %s", e)
            return {}

    @staticmethod
    def _df_to_records(df) -> list[dict]:
        """Convert a price-history DataFrame to a list of records."""
        if df is None or df.empty:
            return []
        try:
            df = df.reset_index()
            df["Date"] = df["Date"].astype(str).str[:10]
            return df.where(df.notna(), None).to_dict(orient="records")
        except Exception as e:
            logger.warning("Price history conversion failed: %s", e)
            return []


# ---------------------------------------------------------------------------
# News Collector
# ---------------------------------------------------------------------------

class NewsCollector:
    """
    Searches DuckDuckGo for recent news about the company.
    Uses the duckduckgo-search library (free, no API key).
    """

    def get_data(self, ticker: str, company_name: str = "") -> dict:
        ticker = ticker.upper().strip()
        result: dict[str, Any] = {
            "ticker": ticker,
            "news_items": [],
            "errors": [],
        }

        queries = [
            f"{company_name or ticker} stock news",
            f"{ticker} earnings results",
            f"{ticker} analyst rating price target",
        ]

        all_items = []
        for query in queries:
            try:
                items = self._search_news(query)
                all_items.extend(items)
                time.sleep(1.0)  # Be polite between queries
            except Exception as e:
                result["errors"].append(f"DDG search failed for '{query}': {e}")
                logger.warning("News search error: %s", e)

        # De-duplicate by URL
        seen_urls: set[str] = set()
        deduped = []
        for item in all_items:
            url = item.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                deduped.append(item)

        result["news_items"] = deduped[:DDG_MAX_RESULTS * len(queries)]
        return result

    def _search_news(self, query: str) -> list[dict]:
        """Run a single DuckDuckGo news search via the library."""
        try:
            with DDGS() as ddgs:
                raw_results = list(ddgs.news(query, max_results=DDG_MAX_RESULTS))
        except Exception as e:
            raise RuntimeError(f"DDG news search failed: {e}") from e

        items = []
        for r in raw_results:
            items.append({
                "title": r.get("title", ""),
                "snippet": r.get("body", ""),
                "url": r.get("url", ""),
                "source": r.get("source", ""),
                "date": r.get("date", ""),
                "query": query,
            })
        return items


# ---------------------------------------------------------------------------
# Earnings Collector
# ---------------------------------------------------------------------------

class EarningsCollector:
    """
    Gathers earnings dates, EPS estimates vs actuals, and 8-K filing links.
    Uses yfinance for structured earnings data and SEC EDGAR for 8-K URLs.
    """

    def get_data(self, ticker: str, sec_collector: SECCollector | None = None) -> dict:
        ticker = ticker.upper().strip()
        result: dict[str, Any] = {
            "ticker": ticker,
            "earnings_history": [],
            "earnings_upcoming": {},
            "analyst_eps_estimates": {},
            "recent_8k_urls": [],
            "errors": [],
        }

        try:
            t = yf.Ticker(ticker)
            info = t.info or {}

            # Upcoming earnings date
            if info.get("earningsTimestamp"):
                ts = info["earningsTimestamp"]
                result["earnings_upcoming"]["date"] = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
            if info.get("earningsDate"):
                # Could be a list or a Timestamp
                ed = info["earningsDate"]
                if isinstance(ed, list) and ed:
                    result["earnings_upcoming"]["date_range"] = [str(d)[:10] for d in ed]

            # EPS estimates
            result["analyst_eps_estimates"] = {
                "eps_current_year": info.get("epsCurrentYear"),
                "eps_next_year": info.get("epsForwardAnnual"),
                "eps_next_quarter": info.get("epsNextQuarterEstimate"),
                "eps_forward": info.get("forwardEps"),
                "earnings_growth": info.get("earningsGrowth"),
                "earnings_quarterly_growth": info.get("earningsQuarterlyGrowth"),
            }

        except Exception as e:
            result["errors"].append(f"yfinance earnings data error: {e}")
            logger.warning("Earnings yfinance error for %s: %s", ticker, e)

        # Earnings history from yfinance
        try:
            t = yf.Ticker(ticker)
            eh = t.earnings_history
            if eh is not None and not eh.empty:
                records = eh.reset_index().tail(16).to_dict(orient="records")
                result["earnings_history"] = [
                    {k: (str(v)[:10] if hasattr(v, "strftime") else v) for k, v in r.items()}
                    for r in records
                ]
        except Exception as e:
            result["errors"].append(f"Earnings history error: {e}")

        # 8-K filing URLs from SEC (if sec_collector provided)
        if sec_collector is not None:
            try:
                cik_data = sec_collector.get_data(ticker)
                cik = cik_data.get("cik")
                if cik:
                    padded_cik = str(cik).zfill(10)
                    submissions = sec_collector._fetch_submissions(padded_cik)
                    result["recent_8k_urls"] = sec_collector._extract_filing_urls(
                        submissions, "8-K", 5
                    )
            except Exception as e:
                result["errors"].append(f"SEC 8-K fetch error: {e}")
                logger.warning("8-K fetch error for %s: %s", ticker, e)

        return result


# ---------------------------------------------------------------------------
# Convenience: collect everything
# ---------------------------------------------------------------------------

def collect_all(ticker: str) -> dict:
    """
    Run all four collectors and return a unified data dict.
    This is the main entry point used by Phase 1 agents.
    """
    ticker = ticker.upper().strip()
    logger.info("Starting data collection for %s …", ticker)

    sec = SECCollector()
    market = MarketDataCollector()
    news = NewsCollector()
    earnings = EarningsCollector()

    sec_data = {}
    market_data = {}
    news_data = {}
    earnings_data = {}

    try:
        logger.info("[%s] Collecting SEC EDGAR data …", ticker)
        sec_data = sec.get_data(ticker)
    except Exception as e:
        logger.error("[%s] SEC collection failed: %s", ticker, e)
        sec_data = {"ticker": ticker, "errors": [str(e)]}

    company_name = sec_data.get("company_name", "") or ""

    try:
        logger.info("[%s] Collecting market data (yfinance) …", ticker)
        market_data = market.get_data(ticker)
        if not company_name:
            company_name = (
                _safe_get(market_data, "company_info", "name", default="") or ""
            )
    except Exception as e:
        logger.error("[%s] Market data collection failed: %s", ticker, e)
        market_data = {"ticker": ticker, "errors": [str(e)]}

    try:
        logger.info("[%s] Collecting news …", ticker)
        news_data = news.get_data(ticker, company_name=company_name)
    except Exception as e:
        logger.error("[%s] News collection failed: %s", ticker, e)
        news_data = {"ticker": ticker, "news_items": [], "errors": [str(e)]}

    try:
        logger.info("[%s] Collecting earnings data …", ticker)
        earnings_data = earnings.get_data(ticker, sec_collector=None)
    except Exception as e:
        logger.error("[%s] Earnings collection failed: %s", ticker, e)
        earnings_data = {"ticker": ticker, "errors": [str(e)]}

    logger.info("[%s] Data collection complete.", ticker)

    return {
        "ticker": ticker,
        "company_name": company_name,
        "collected_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sec": sec_data,
        "market": market_data,
        "news": news_data,
        "earnings": earnings_data,
    }
