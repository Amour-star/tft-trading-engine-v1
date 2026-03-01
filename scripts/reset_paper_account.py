"""CLI helper to nuke and rebuild the paper account."""

from __future__ import annotations

import argparse
import json
import sys

from loguru import logger

from paper.reset import reset_paper_account


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reset the paper account ledger, predictions, and metrics."
    )
    parser.add_argument(
        "--initial-balance",
        type=float,
        help="Target paper wallet balance (defaults to PAPER_INITIAL_BALANCE or configured starting balance)",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Apply the reset. Without this flag the command performs a dry run.",
    )
    args = parser.parse_args()

    if not args.confirm:
        logger.info("Dry run: no data will be modified. Add --confirm to execute.")

    try:
        summary = reset_paper_account(
            initial_balance=args.initial_balance,
            confirm=args.confirm,
        )
    except Exception:
        logger.exception("Paper reset failed")
        sys.exit(1)

    payload = summary.to_dict()
    print(json.dumps(payload, indent=2))

    if args.confirm:
        logger.success(
            "Paper account reset applied (initial balance=%s).",
            payload["initial_balance"],
        )
    else:
        logger.info("Dry run complete. No changes were committed.")


if __name__ == "__main__":
    main()
