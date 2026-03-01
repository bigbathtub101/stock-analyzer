#!/usr/bin/env python3
"""Deep Equity Research System - Run analysis for any stock ticker.

Usage:
    python run.py AAPL
    python run.py MSFT --parallel
    python run.py NVDA --debug
"""
from orchestrator import main

if __name__ == "__main__":
    main()
