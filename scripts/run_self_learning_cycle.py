"""
Run the daily self-learning loop: analyze trades, retrain models, and persist a report.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.database import init_db
from engine.feedback import FeedbackLoop
from utils.logging import setup_logging


def _run_subprocess(args: list[str]) -> dict:
    started = datetime.utcnow().isoformat()
    try:
        completed = subprocess.run(args, cwd=str(ROOT), capture_output=True, text=True, check=False)
        return {
            "started_at": started,
            "returncode": int(completed.returncode),
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
        }
    except Exception as exc:
        return {
            "started_at": started,
            "returncode": -1,
            "stdout_tail": "",
            "stderr_tail": str(exc),
        }


def run_cycle(args: argparse.Namespace) -> dict:
    init_db()
    feedback = FeedbackLoop()
    stats = feedback.compute_trade_statistics(limit=args.trade_sample)
    adjustments = feedback.compute_batch_adjustments()

    required_symbols = list(dict.fromkeys(args.symbols))
    tasks: dict[str, dict] = {}

    if not args.skip_tft:
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "train_model.py"),
            "--timeframe",
            args.timeframe,
            "--symbols",
            *required_symbols,
            "--required-symbols",
            *required_symbols,
        ]
        if args.quick:
            cmd.append("--quick")
        tasks["tft_training"] = _run_subprocess(cmd)

    if not args.skip_meta:
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "train_meta_model.py"),
            "--limit",
            str(args.meta_limit),
            "--walk-forward-folds",
            str(args.walk_forward_folds),
        ]
        tasks["meta_training"] = _run_subprocess(cmd)

    if not args.skip_rl:
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "train_rl_manager.py"),
        ]
        tasks["rl_training"] = _run_subprocess(cmd)

    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "timeframe": args.timeframe,
        "symbols": required_symbols,
        "trade_statistics": stats,
        "threshold_adjustments": adjustments,
        "tasks": tasks,
    }

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Self-learning cycle report written to {}", report_path)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the self-learning retrain cycle")
    parser.add_argument("--timeframe", default="15min")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTC-USDT", "ETH-USDT", "XRP-USDT", "DOGE-USDT"],
    )
    parser.add_argument("--trade-sample", type=int, default=500)
    parser.add_argument("--meta-limit", type=int, default=5000)
    parser.add_argument("--walk-forward-folds", type=int, default=4)
    parser.add_argument("--report-path", default=str(ROOT / "state" / "reports" / "self_learning_latest.json"))
    parser.add_argument("--interval-hours", type=float, default=24.0)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--skip-tft", action="store_true")
    parser.add_argument("--skip-meta", action="store_true")
    parser.add_argument("--skip-rl", action="store_true")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    while True:
        run_cycle(args)
        if args.once:
            return
        time.sleep(max(3600.0, float(args.interval_hours) * 3600.0))


if __name__ == "__main__":
    main()
