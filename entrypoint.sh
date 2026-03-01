#!/usr/bin/env bash
set -e

echo "[ENTRYPOINT] Checking engine state..."

if [ "$ENGINE_MODE" == "PAPER" ]; then
echo "PAPER mode: resetting positions DB if first start"
if [ ! -f /app/state/.paper_init_done ]; then
rm -f /app/state/tft_engine.db
touch /app/state/.paper_init_done
fi
fi

echo "[ENTRYPOINT] Starting engine..."
exec python3 /app/scripts/run_engine.py "$@"