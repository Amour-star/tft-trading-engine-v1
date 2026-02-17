#!/usr/bin/env python3
"""
Quick KuCoin API connectivity test.

Used by verify.py and fix_issues.py to validate:
1) Public market API access.
2) Authenticated account API access.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings
from data.fetcher import KuCoinDataFetcher


def _is_placeholder(value: str) -> bool:
    v = (value or "").strip().lower()
    return not v or "your_" in v or "changeme" in v


def main() -> int:
    print("=" * 60)
    print("KuCoin API Connection Test")
    print("=" * 60)

    cfg = settings.kucoin
    missing = []
    if _is_placeholder(cfg.api_key):
        missing.append("KUCOIN_API_KEY")
    if _is_placeholder(cfg.api_secret):
        missing.append("KUCOIN_API_SECRET")
    if _is_placeholder(cfg.api_passphrase):
        missing.append("KUCOIN_API_PASSPHRASE")

    if missing:
        print(f"Missing or placeholder credentials: {', '.join(missing)}")
        return 1

    fetcher = KuCoinDataFetcher()

    # Public endpoint sanity check
    try:
        ticker = fetcher.market.get_ticker("BTC-USDT")
        price = float(ticker.get("price", 0))
        if price <= 0:
            raise ValueError("invalid ticker price returned")
        print(f"Public API: OK (BTC-USDT price={price:.6f})")
    except Exception as e:
        print(f"Public API: FAILED ({e})")
        return 1

    # Private endpoint auth check
    try:
        accounts = fetcher.user_client.get_account_list(account_type="trade")
        if not isinstance(accounts, list):
            raise RuntimeError(f"unexpected account response type: {type(accounts).__name__}")

        non_zero = 0
        total_value = 0.0
        for acc in accounts:
            available = float(acc.get("available", 0) or 0)
            holds = float(acc.get("holds", 0) or 0)
            balance = available + holds
            total_value += balance
            if balance > 0:
                non_zero += 1

        print("Account summary:")
        print(f"  Trade accounts: {len(accounts)}")
        print(f"  Non-zero balances: {non_zero}")
        print(f"  Total value (raw sum): {total_value:.8f}")
    except Exception as e:
        print(f"Private API: FAILED ({e})")
        return 1

    print("All tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

