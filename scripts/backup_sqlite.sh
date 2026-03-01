#!/usr/bin/env bash
set -euo pipefail

# Run via cron every 6 hours:
# 0 */6 * * * /home/ubuntu/tft/scripts/backup_sqlite.sh >> /home/ubuntu/tft/logs/backup_sqlite.log 2>&1

DB_PATH="${SQLITE_PATH:-/home/ubuntu/tft/data/tft_engine.db}"
BACKUP_DIR="${SQLITE_BACKUP_DIR:-/home/ubuntu/tft/data/backups}"
RETENTION_DAYS="${SQLITE_BACKUP_RETENTION_DAYS:-14}"

mkdir -p "$BACKUP_DIR"

timestamp="$(date +%Y%m%d_%H%M%S)"
backup_file="$BACKUP_DIR/tft_engine_${timestamp}.db"

sqlite3 "$DB_PATH" ".backup '$backup_file'"

find "$BACKUP_DIR" -type f -name "tft_engine_*.db" -mtime +"$RETENTION_DAYS" -delete

echo "backup_complete db=$DB_PATH file=$backup_file retention_days=$RETENTION_DAYS"
