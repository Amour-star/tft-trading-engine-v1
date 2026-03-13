"""End-to-end smoke test for TFT engine APIs + dashboard."""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List

import requests


DEFAULT_APIS: Dict[str, str] = {
    "BTC-USDT": "http://localhost:8001/api",
    "ETH-USDT": "http://localhost:8002/api",
    "XRP-USDT": "http://localhost:8003/api",
    "DOGE-USDT": "http://localhost:8004/api",
}


@dataclass
class SmokeResult:
    symbol: str
    api_base: str
    status_ok: bool = False
    metrics_ok: bool = False
    trades_ok: bool = False
    positions_ok: bool = False
    equity_ok: bool = False
    market_data_source: str = "unknown"
    synthetic_active: bool = False
    trade_count: int = 0


def _get_json(url: str, params: Dict[str, Any] | None = None, timeout_sec: float = 8.0) -> Any:
    response = requests.get(url, params=params, timeout=timeout_sec)
    response.raise_for_status()
    return response.json()


def _post_json(url: str, payload: Dict[str, Any], timeout_sec: float = 8.0) -> Any:
    response = requests.post(url, json=payload, timeout=timeout_sec)
    response.raise_for_status()
    return response.json()


def _assert_finite_number(value: Any, label: str) -> None:
    if value is None:
        raise AssertionError(f"{label} is None")
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise AssertionError(f"{label} is NaN/Inf")
        return
    raise AssertionError(f"{label} is not numeric: {type(value).__name__}")


def _parse_api_map(raw: str) -> Dict[str, str]:
    if not raw.strip():
        return dict(DEFAULT_APIS)
    result: Dict[str, str] = {}
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"Invalid --apis token: {token}")
        symbol, base = token.split("=", 1)
        result[symbol.strip().upper()] = base.strip().rstrip("/")
    return result


def _validate_symbol_api(symbol: str, api_base: str) -> SmokeResult:
    result = SmokeResult(symbol=symbol, api_base=api_base)

    status = _get_json(f"{api_base}/status")
    metrics = _get_json(f"{api_base}/metrics")
    trades = _get_json(f"{api_base}/trades", params={"limit": 20})
    positions = _get_json(f"{api_base}/positions")
    equity = _get_json(f"{api_base}/equity", params={"limit": 60})
    performance = _get_json(f"{api_base}/performance")

    for key in ("mode", "engine_running", "trading_enabled", "market_data_source"):
        if key not in status:
            raise AssertionError(f"{symbol} status missing key: {key}")
    result.market_data_source = str(status.get("market_data_source", "unknown"))
    result.synthetic_active = bool(status.get("synthetic_active", False))
    result.status_ok = True

    for key in (
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "win_rate",
        "profit_factor",
        "average_trade",
        "exposure_pct",
        "rolling_volatility",
        "total_trades",
    ):
        _assert_finite_number(metrics.get(key), f"{symbol} metrics.{key}")
    result.metrics_ok = True

    if not isinstance(trades, list):
        raise AssertionError(f"{symbol} trades payload is not a list")
    for idx, trade in enumerate(trades):
        for key in ("entry_price", "stop_price", "target_price", "pnl", "pnl_pct", "confidence"):
            _assert_finite_number(trade.get(key), f"{symbol} trades[{idx}].{key}")
    result.trade_count = len(trades)
    result.trades_ok = True

    if not isinstance(positions, list):
        raise AssertionError(f"{symbol} positions payload is not a list")
    for idx, position in enumerate(positions):
        for key in ("quantity", "entry_price", "mark_price", "unrealized_pnl"):
            _assert_finite_number(position.get(key), f"{symbol} positions[{idx}].{key}")
    result.positions_ok = True

    if not isinstance(equity, list) or not equity:
        raise AssertionError(f"{symbol} equity payload is empty")
    for idx, row in enumerate(equity[:10]):
        for key in ("equity", "balance", "realized_pnl", "unrealized_pnl", "open_positions"):
            _assert_finite_number(row.get(key), f"{symbol} equity[{idx}].{key}")
    result.equity_ok = True

    # Basic consistency check between /status and /performance snapshots.
    status_equity = float(status.get("paper_equity") or performance.get("equity") or 0.0)
    perf_equity = float(performance.get("equity") or 0.0)
    if max(abs(status_equity), abs(perf_equity), 1.0) > 0:
        delta_pct = abs(status_equity - perf_equity) / max(abs(status_equity), abs(perf_equity), 1.0)
        if delta_pct > 0.1:
            raise AssertionError(
                f"{symbol} status/performance equity mismatch too large "
                f"(status={status_equity}, performance={perf_equity})"
            )

    return result


def _wait_for_trade(api_map: Dict[str, str], timeout_sec: int) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        for symbol, api_base in api_map.items():
            trades = _get_json(f"{api_base}/trades", params={"limit": 5})
            if isinstance(trades, list) and len(trades) > 0:
                return
        time.sleep(5)
    diagnostics: Dict[str, Dict[str, Any]] = {}
    for symbol, api_base in api_map.items():
        try:
            diagnostics[symbol] = {
                "status": _get_json(f"{api_base}/status"),
                "engine_state": _get_json(f"{api_base}/engine-state"),
                "decision_events": _get_json(f"{api_base}/decision-events", params={"limit": 3}),
            }
        except Exception as exc:
            diagnostics[symbol] = {"error": str(exc)}
    raise AssertionError(
        f"No trades observed within {timeout_sec}s. Diagnostics: {json.dumps(diagnostics, default=str)}"
    )


def _resume_if_paused(api_map: Dict[str, str]) -> None:
    for _symbol, api_base in api_map.items():
        status = _get_json(f"{api_base}/status")
        if bool(status.get("paused", False)):
            _post_json(f"{api_base}/control", {"action": "resume"})


def _validate_dashboard(dashboard_url: str) -> None:
    health = requests.get(f"{dashboard_url.rstrip('/')}/_stcore/health", timeout=8)
    health.raise_for_status()
    root = requests.get(dashboard_url.rstrip("/"), timeout=8)
    root.raise_for_status()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run smoke E2E checks against TFT APIs and dashboard.")
    parser.add_argument(
        "--apis",
        default="",
        help="Comma-separated SYMBOL=http://host:port/api map. Default checks BTC/ETH/XRP/DOGE local APIs.",
    )
    parser.add_argument(
        "--dashboard-url",
        default="http://localhost:8511",
        help="Dashboard base URL (default: http://localhost:8511).",
    )
    parser.add_argument(
        "--wait-trade-timeout",
        type=int,
        default=300,
        help="Seconds to wait for at least one paper trade before failing (default: 300).",
    )
    parser.add_argument(
        "--skip-trade-wait",
        action="store_true",
        help="Skip waiting for at least one trade.",
    )
    args = parser.parse_args()

    api_map = _parse_api_map(args.apis)
    if not api_map:
        raise SystemExit("No API endpoints configured.")

    _resume_if_paused(api_map)

    if not args.skip_trade_wait:
        _wait_for_trade(api_map, timeout_sec=max(1, int(args.wait_trade_timeout)))

    results: List[SmokeResult] = []
    for symbol, api_base in api_map.items():
        results.append(_validate_symbol_api(symbol, api_base.rstrip("/")))

    _validate_dashboard(args.dashboard_url)

    synthetic_symbols = [r.symbol for r in results if r.synthetic_active]
    summary = {
        "checked_symbols": [r.symbol for r in results],
        "synthetic_active_symbols": synthetic_symbols,
        "results": [r.__dict__ for r in results],
        "dashboard_ok": True,
    }
    print(json.dumps(summary, indent=2))
    if synthetic_symbols:
        print("WARNING: synthetic market data active for:", ", ".join(synthetic_symbols))


if __name__ == "__main__":
    main()
