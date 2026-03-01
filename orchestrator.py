from __future__ import annotations

"""
Main orchestrator for the Deep Equity Research System.

Runs all 7 phases and 27 agents sequentially (or 2-at-a-time with --parallel flag),
saves full outputs and compressed briefings, and assembles the final report.

Usage:
    python orchestrator.py TICKER [--parallel]
    python run.py TICKER [--parallel]
"""

import argparse
import json
import logging
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from config import (
    OUTPUT_DIR,
    PHASE_CONFIG,
    TOTAL_AGENTS,
    TEMP_ANALYSIS,
    TEMP_CREATIVE,
    MAX_TOKENS_FULL,
)
from data_collectors import collect_all
from llm_client import GeminiClient
import prompts

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _elapsed(start: float) -> str:
    s = time.time() - start
    if s < 60:
        return f"{s:.0f}s"
    return f"{s/60:.1f}min"


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _read_file(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def _banner(ticker: str, parallel: bool) -> None:
    print()
    print("=" * 70)
    print("    DEEP EQUITY RESEARCH SYSTEM")
    print(f"    Ticker: {ticker.upper()}")
    print(f"    Agents: {TOTAL_AGENTS} across 7 phases")
    print(f"    Mode:   {'Parallel (2 agents at a time)' if parallel else 'Sequential'}")
    print(f"    Model:  gemini-2.5-flash (free tier)")
    print()
    est_low = TOTAL_AGENTS * 8 // 60
    est_high = est_low + 20
    if parallel:
        est_low = est_low // 2
        est_high = est_high // 2
    print(f"    Estimated time: {est_low}–{est_high} minutes")
    print(f"    Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()


def _phase_header(phase_num: int, phase_name: str, agent_count: int) -> None:
    print(f"\n{'─'*60}")
    print(f"  PHASE {phase_num}: {phase_name.upper()}")
    print(f"  {agent_count} agent{'s' if agent_count > 1 else ''}")
    print(f"{'─'*60}")


def _agent_line(agent_id: int, agent_name: str, status: str, elapsed: str = "") -> None:
    suffix = f"  [{elapsed}]" if elapsed else ""
    print(f"  [{_now()}]  Agent {agent_id:02d} — {agent_name:<35} {status}{suffix}")


# ---------------------------------------------------------------------------
# Agent Runner
# ---------------------------------------------------------------------------

class AgentRunner:
    """Executes a single agent: builds prompt, calls LLM, saves outputs."""

    def __init__(self, client: GeminiClient, output_dir: Path):
        self.client = client
        self.output_dir = output_dir

    def run(
        self,
        phase_num: int,
        agent_cfg: dict,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, str]:
        """
        Execute one agent.

        Returns
        -------
        tuple[str, str]
            (full_output, briefing)
        """
        agent_id = agent_cfg["id"]
        agent_name = agent_cfg["name"]
        temp = temperature if temperature is not None else agent_cfg.get("temp", TEMP_ANALYSIS)
        tokens = max_tokens if max_tokens is not None else agent_cfg.get("max_tokens", MAX_TOKENS_FULL)

        t0 = time.time()
        _agent_line(agent_id, agent_name, "⟳ running")

        try:
            full_output, briefing = self.client.generate_with_briefing(
                prompt=prompt,
                temperature=temp,
                max_tokens=tokens,
            )
        except Exception as exc:
            err_msg = f"ERROR in Agent {agent_id} ({agent_name}): {traceback.format_exc()}"
            logger.error(err_msg)
            full_output = f"# Agent {agent_id}: {agent_name}\n\n**ERROR:** {exc}\n\nPlease check logs."
            briefing = f"Agent {agent_id} ({agent_name}) failed with error: {exc}"

        # Save outputs
        full_path = self.output_dir / f"phase{phase_num}_agent{agent_id:02d}_full.md"
        brief_path = self.output_dir / f"phase{phase_num}_agent{agent_id:02d}_briefing.md"
        _write_file(full_path, f"# Agent {agent_id}: {agent_name}\n\n{full_output}")
        _write_file(brief_path, f"# Agent {agent_id}: {agent_name} — Briefing\n\n{briefing}")

        elapsed = _elapsed(t0)
        _agent_line(agent_id, agent_name, "✓ done", elapsed)
        return full_output, briefing


# ---------------------------------------------------------------------------
# Phase Executors
# ---------------------------------------------------------------------------

def run_phase_sequential(
    runner: AgentRunner,
    phase_num: int,
    agents_and_prompts: list[tuple[dict, str]],
) -> dict[int, tuple[str, str]]:
    """Run agents one after another."""
    results = {}
    for agent_cfg, prompt in agents_and_prompts:
        full, brief = runner.run(phase_num, agent_cfg, prompt)
        results[agent_cfg["id"]] = (full, brief)
    return results


def run_phase_parallel_2(
    runner: AgentRunner,
    phase_num: int,
    agents_and_prompts: list[tuple[dict, str]],
) -> dict[int, tuple[str, str]]:
    """
    Run agents in batches of 2 (to stay within free tier limits).
    Uses threading so both calls run concurrently per batch.
    """
    import threading

    results: dict[int, tuple[str, str]] = {}
    pairs = [agents_and_prompts[i:i+2] for i in range(0, len(agents_and_prompts), 2)]

    for pair in pairs:
        batch_results: dict[int, tuple[str, str]] = {}
        errors: dict[int, Exception] = {}

        def _run_one(cfg, prm):
            try:
                f, b = runner.run(phase_num, cfg, prm)
                batch_results[cfg["id"]] = (f, b)
            except Exception as e:
                errors[cfg["id"]] = e
                batch_results[cfg["id"]] = (
                    f"ERROR: {e}",
                    f"Agent {cfg['id']} failed: {e}",
                )

        threads = [threading.Thread(target=_run_one, args=(c, p), daemon=True) for c, p in pair]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        results.update(batch_results)
        if errors:
            for aid, err in errors.items():
                logger.warning("Batch error agent %d: %s", aid, err)

    return results


def combine_briefings(
    output_dir: Path,
    phase_num: int,
    agent_results: dict[int, tuple[str, str]],
    agent_names: dict[int, str],
) -> str:
    """Concatenate all agent briefings for a phase and save combined file."""
    combined_parts = [f"# Phase {phase_num} Combined Briefings\n"]
    for agent_id in sorted(agent_results.keys()):
        _, briefing = agent_results[agent_id]
        name = agent_names.get(agent_id, f"Agent {agent_id}")
        combined_parts.append(f"\n## Agent {agent_id}: {name}\n\n{briefing}\n")

    combined = "\n".join(combined_parts)
    path = output_dir / f"phase{phase_num}_combined_briefing.md"
    _write_file(path, combined)
    return combined


# ---------------------------------------------------------------------------
# Main Orchestration
# ---------------------------------------------------------------------------

class Orchestrator:

    def __init__(self, ticker: str, parallel: bool = False):
        self.ticker = ticker.upper().strip()
        self.parallel = parallel
        self.client = GeminiClient()
        # Create output directory: output/{TICKER}_{YYYYMMDD_HHMMSS}/
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = OUTPUT_DIR / f"{self.ticker}_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.runner = AgentRunner(self.client, self.output_dir)

        # Will be populated during execution
        self.raw_data: dict = {}
        self.phase_briefings: dict[int, str] = {}
        self.phase_results: dict[int, dict[int, tuple[str, str]]] = {}

        logger.info("Output directory: %s", self.output_dir)

    def run(self) -> Path:
        """Execute all 7 phases and return the path to the final report."""
        t_start = time.time()

        # ── STEP 0: Collect Data ──────────────────────────────────────────
        print(f"\n[{_now()}]  Collecting market data for {self.ticker} …")
        t_collect = time.time()
        try:
            self.raw_data = collect_all(self.ticker)
        except Exception as e:
            logger.error("Data collection failed: %s", e)
            self.raw_data = {
                "ticker": self.ticker,
                "company_name": self.ticker,
                "sec": {}, "market": {}, "news": {}, "earnings": {},
            }
        company_name = self.raw_data.get("company_name") or self.ticker
        print(f"[{_now()}]  Data collected for {company_name} [{_elapsed(t_collect)}]")

        # Save raw data snapshot
        _write_file(
            self.output_dir / "raw_data.json",
            json.dumps(self.raw_data, default=str, indent=2)[:2_000_000],  # cap at 2MB
        )

        # ── PHASES ───────────────────────────────────────────────────────
        self._run_phase_1(company_name)
        self._run_phase_2(company_name)
        self._run_phase_3(company_name)
        self._run_phase_4(company_name)
        self._run_phase_5(company_name)
        self._run_phase_6(company_name)
        self._run_phase_7(company_name)

        # ── FINAL REPORT ──────────────────────────────────────────────────
        final_path = self._assemble_final_report(company_name)

        total = _elapsed(t_start)
        print()
        print("=" * 70)
        print(f"  ANALYSIS COMPLETE  —  {self.ticker}  —  {total}")
        print(f"  Output: {self.output_dir}")
        print(f"  Final report: {final_path.name}")
        print(f"  API calls today: {self.client.requests_today}")
        print("=" * 70)
        print()

        return final_path

    # ------------------------------------------------------------------
    # Phase implementations
    # ------------------------------------------------------------------

    def _get_briefings_up_to(self, up_to_phase: int) -> str:
        """Concatenate all phase combined briefings through up_to_phase."""
        parts = []
        for p in range(1, up_to_phase + 1):
            if p in self.phase_briefings:
                parts.append(f"=== PHASE {p} BRIEFINGS ===\n\n{self.phase_briefings[p]}")
        return "\n\n".join(parts)

    def _run_and_record_phase(self, phase_num: int, agents_and_prompts: list[tuple[dict, str]]) -> None:
        """Execute a phase (sequential or parallel) and save combined briefings."""
        t0 = time.time()
        phase_cfg = PHASE_CONFIG[phase_num]
        agent_names = {a["id"]: a["name"] for a in phase_cfg["agents"]}

        _phase_header(phase_num, phase_cfg["name"], len(agents_and_prompts))

        if self.parallel:
            results = run_phase_parallel_2(self.runner, phase_num, agents_and_prompts)
        else:
            results = run_phase_sequential(self.runner, phase_num, agents_and_prompts)

        combined = combine_briefings(self.output_dir, phase_num, results, agent_names)
        self.phase_briefings[phase_num] = combined
        self.phase_results[phase_num] = results

        print(f"  Phase {phase_num} complete [{_elapsed(t0)}]")

    def _run_phase_1(self, company_name: str) -> None:
        sec_data = self.raw_data.get("sec", {})
        market_data = self.raw_data.get("market", {})
        news_data = self.raw_data.get("news", {})
        earnings_data = self.raw_data.get("earnings", {})

        agents_and_prompts = [
            (
                PHASE_CONFIG[1]["agents"][0],
                prompts.p1_sec_filing_analyst(self.ticker, company_name, sec_data, market_data),
            ),
            (
                PHASE_CONFIG[1]["agents"][1],
                prompts.p1_earnings_call_analyst(self.ticker, company_name, earnings_data, market_data),
            ),
            (
                PHASE_CONFIG[1]["agents"][2],
                prompts.p1_market_data_analyst(self.ticker, company_name, market_data),
            ),
            (
                PHASE_CONFIG[1]["agents"][3],
                prompts.p1_news_sentiment_analyst(self.ticker, company_name, news_data, market_data),
            ),
        ]
        self._run_and_record_phase(1, agents_and_prompts)

    def _run_phase_2(self, company_name: str) -> None:
        p1_briefings = self._get_briefings_up_to(1)

        agents_and_prompts = [
            (
                PHASE_CONFIG[2]["agents"][0],
                prompts.p2_income_statement_analyst(self.ticker, company_name, p1_briefings, self.raw_data),
            ),
            (
                PHASE_CONFIG[2]["agents"][1],
                prompts.p2_balance_sheet_analyst(self.ticker, company_name, p1_briefings, self.raw_data),
            ),
            (
                PHASE_CONFIG[2]["agents"][2],
                prompts.p2_cash_flow_analyst(self.ticker, company_name, p1_briefings, self.raw_data),
            ),
            (
                PHASE_CONFIG[2]["agents"][3],
                prompts.p2_segment_analyst(self.ticker, company_name, p1_briefings, self.raw_data),
            ),
            (
                PHASE_CONFIG[2]["agents"][4],
                prompts.p2_historical_trends_analyst(self.ticker, company_name, p1_briefings, self.raw_data),
            ),
        ]
        self._run_and_record_phase(2, agents_and_prompts)

    def _run_phase_3(self, company_name: str) -> None:
        p1_2_briefings = self._get_briefings_up_to(2)

        agents_and_prompts = [
            (
                PHASE_CONFIG[3]["agents"][0],
                prompts.p3_management_governance(self.ticker, company_name, p1_2_briefings, self.raw_data),
            ),
            (
                PHASE_CONFIG[3]["agents"][1],
                prompts.p3_competitive_moat_assessment(self.ticker, company_name, p1_2_briefings, self.raw_data),
            ),
            (
                PHASE_CONFIG[3]["agents"][2],
                prompts.p3_industry_tam_analysis(self.ticker, company_name, p1_2_briefings, self.raw_data),
            ),
            (
                PHASE_CONFIG[3]["agents"][3],
                prompts.p3_risk_factor_assessment(self.ticker, company_name, p1_2_briefings, self.raw_data),
            ),
        ]
        self._run_and_record_phase(3, agents_and_prompts)

    def _run_phase_4(self, company_name: str) -> None:
        p1_3_briefings = self._get_briefings_up_to(3)

        agents_and_prompts = [
            (
                PHASE_CONFIG[4]["agents"][0],
                prompts.p4_dcf_valuation(self.ticker, company_name, p1_3_briefings, self.raw_data),
            ),
            (
                PHASE_CONFIG[4]["agents"][1],
                prompts.p4_comparable_company(self.ticker, company_name, p1_3_briefings, self.raw_data),
            ),
            (
                PHASE_CONFIG[4]["agents"][2],
                prompts.p4_historical_valuation(self.ticker, company_name, p1_3_briefings, self.raw_data),
            ),
            (
                PHASE_CONFIG[4]["agents"][3],
                prompts.p4_sotp_valuation(self.ticker, company_name, p1_3_briefings, self.raw_data),
            ),
            (
                PHASE_CONFIG[4]["agents"][4],
                prompts.p4_revenue_evolution_model(self.ticker, company_name, p1_3_briefings, self.raw_data),
            ),
        ]
        self._run_and_record_phase(4, agents_and_prompts)

    def _run_phase_5(self, company_name: str) -> None:
        p1_4_briefings = self._get_briefings_up_to(4)

        agents_and_prompts = [
            (
                PHASE_CONFIG[5]["agents"][0],
                prompts.p5_sector_vulnerability_mapper(self.ticker, company_name, p1_4_briefings, self.raw_data),
            ),
            (
                PHASE_CONFIG[5]["agents"][1],
                prompts.p5_competitive_threat_tracker(self.ticker, company_name, p1_4_briefings, self.raw_data),
            ),
            (
                PHASE_CONFIG[5]["agents"][2],
                prompts.p5_scenario_progression_mapper(self.ticker, company_name, p1_4_briefings, self.raw_data),
            ),
            (
                PHASE_CONFIG[5]["agents"][3],
                prompts.p5_thesis_narrative_builder(self.ticker, company_name, p1_4_briefings, self.raw_data),
            ),
        ]
        self._run_and_record_phase(5, agents_and_prompts)

    def _run_phase_6(self, company_name: str) -> None:
        all_prior_briefings = self._get_briefings_up_to(5)

        # Pass Phase 5 full outputs to synthesis agents
        phase5_results = self.phase_results.get(5, {})
        phase5_full_combined = "\n\n".join(
            f"### Agent {aid} Full Output\n\n{full}"
            for aid, (full, _) in sorted(phase5_results.items())
        )

        # Use the most important Phase 4 valuation details for the synthesis agent
        phase4_results = self.phase_results.get(4, {})
        valuation_details = "\n\n".join(
            f"### Agent {aid} Output\n\n{full[:1500]}"
            for aid, (full, _) in sorted(phase4_results.items())
        )

        agents_and_prompts = [
            (
                PHASE_CONFIG[6]["agents"][0],
                prompts.p6_bull_case_builder(self.ticker, company_name, all_prior_briefings, phase5_full_combined),
            ),
            (
                PHASE_CONFIG[6]["agents"][1],
                prompts.p6_bear_case_builder(self.ticker, company_name, all_prior_briefings, phase5_full_combined),
            ),
            (
                PHASE_CONFIG[6]["agents"][2],
                prompts.p6_final_synthesis(self.ticker, company_name, all_prior_briefings, valuation_details),
            ),
        ]
        self._run_and_record_phase(6, agents_and_prompts)

    def _run_phase_7(self, company_name: str) -> None:
        all_briefings = self._get_briefings_up_to(6)

        # Give Agent 26 the full Phase 6 synthesis
        phase6_results = self.phase_results.get(6, {})
        # Agent 25 is the Final Synthesis (id=25) — find it
        synthesis_full = ""
        for aid, (full, _) in phase6_results.items():
            if aid == 25:
                synthesis_full = full
                break
        if not synthesis_full and phase6_results:
            # Fall back to last agent output
            synthesis_full = list(phase6_results.values())[-1][0]

        agents_and_prompts = [
            (
                PHASE_CONFIG[7]["agents"][0],
                prompts.p7_evidence_based_rating(self.ticker, company_name, all_briefings, synthesis_full),
            ),
        ]
        self._run_and_record_phase(7, agents_and_prompts)

    # ------------------------------------------------------------------
    # Final report assembly
    # ------------------------------------------------------------------

    def _assemble_final_report(self, company_name: str) -> Path:
        print(f"\n[{_now()}]  Assembling final report …")

        sections = []
        sections.append(f"# Deep Equity Research Report: {company_name} ({self.ticker})")
        sections.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        sections.append(f"*System: Deep Equity Research System — 26 agents, 7 phases*")
        sections.append("")
        sections.append("---")
        sections.append("")

        # Table of contents
        sections.append("## Table of Contents")
        sections.append("1. Executive Summary (Phase 7 Academic Rating)")
        sections.append("2. Final Synthesis (Phase 6)")
        sections.append("3. Bull Case (Phase 6)")
        sections.append("4. Bear Case (Phase 6)")
        sections.append("5. Forward Intelligence (Phase 5)")
        sections.append("6. Valuation (Phase 4)")
        sections.append("7. Qualitative Analysis (Phase 3)")
        sections.append("8. Financial Analysis (Phase 2)")
        sections.append("9. Data Analysis (Phase 1)")
        sections.append("")
        sections.append("---")
        sections.append("")

        # Ordered phases for report (reverse order — most important first)
        report_order = [
            (7, "Executive Summary — Academic Evidence Rating"),
            (6, "Final Investment Synthesis"),
            (5, "Forward Intelligence"),
            (4, "Valuation Analysis"),
            (3, "Qualitative Analysis"),
            (2, "Financial Analysis"),
            (1, "Data Analysis"),
        ]

        for phase_num, phase_title in report_order:
            sections.append(f"## Phase {phase_num}: {phase_title}")
            sections.append("")

            phase_results = self.phase_results.get(phase_num, {})
            phase_cfg = PHASE_CONFIG.get(phase_num, {})
            agent_names = {a["id"]: a["name"] for a in phase_cfg.get("agents", [])}

            for agent_id in sorted(phase_results.keys()):
                full_output, _ = phase_results[agent_id]
                name = agent_names.get(agent_id, f"Agent {agent_id}")
                sections.append(f"### Agent {agent_id}: {name}")
                sections.append("")
                sections.append(full_output)
                sections.append("")
                sections.append("---")
                sections.append("")

        report = "\n".join(sections)
        final_path = self.output_dir / "final_report.md"
        _write_file(final_path, report)
        print(f"[{_now()}]  Final report saved: {final_path}")
        return final_path


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deep Equity Research System — 26-agent, 7-phase stock analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python orchestrator.py AAPL
  python orchestrator.py MSFT --parallel
  python run.py NVDA

Get a free Gemini API key at https://aistudio.google.com/apikey
Set it: export GEMINI_API_KEY=your_key_here
        """,
    )
    parser.add_argument(
        "ticker",
        type=str,
        help="Stock ticker symbol (e.g., AAPL, MSFT, NVDA)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=False,
        help="Run 2 agents at a time per phase (faster but still respects rate limits)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug-level logging",
    )
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    ticker = args.ticker.upper().strip()
    if not ticker.isalpha() or len(ticker) > 6:
        print(f"Error: '{ticker}' does not look like a valid stock ticker.")
        sys.exit(1)

    _banner(ticker, args.parallel)

    orchestrator = Orchestrator(ticker=ticker, parallel=args.parallel)
    try:
        final_path = orchestrator.run()
        print(f"\nYour report: {final_path}\n")
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user. Partial results saved in:")
        print(f"  {orchestrator.output_dir}\n")
        sys.exit(0)
    except Exception as e:
        logger.error("Analysis failed: %s", traceback.format_exc())
        print(f"\nError: {e}")
        print("Partial results may be saved in:", orchestrator.output_dir)
        sys.exit(1)


if __name__ == "__main__":
    main()
