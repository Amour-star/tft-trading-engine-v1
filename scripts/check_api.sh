#!/usr/bin/env sh
# Usage:
#   sh scripts/check_api.sh [--base-url http://127.0.0.1:8000] [--timeout 1.5]
#                           [--freshness-warn-seconds 120] [--no-discovery]
#
# Purpose:
#   Read-only API/dashboard diagnostics with JSON output for automation.
#   This script never modifies trading state.

set -u

BASE_URL="${API_BASE_URL:-}"
if [ -z "$BASE_URL" ]; then
  api_host="${API_HOST:-127.0.0.1}"
  api_port="${API_PORT:-8000}"
  BASE_URL="http://${api_host}:${api_port}"
fi

TIMEOUT_SECONDS=1.5
FRESHNESS_WARN_SECONDS=120
DISCOVERY=1

json_escape() {
  printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g; s/\r//g; :a;N;$!ba;s/\n/\\n/g'
}

ISSUES_ITEMS=""
ERRORS_ITEMS=""

add_issue() {
  esc=$(json_escape "$1")
  ISSUES_ITEMS="$ISSUES_ITEMS,\"$esc\""
}

add_error() {
  esc=$(json_escape "$1")
  ERRORS_ITEMS="$ERRORS_ITEMS,\"$esc\""
}

while [ $# -gt 0 ]; do
  case "$1" in
    --base-url)
      BASE_URL="${2:-$BASE_URL}"
      shift 2
      ;;
    --timeout)
      TIMEOUT_SECONDS="${2:-$TIMEOUT_SECONDS}"
      shift 2
      ;;
    --freshness-warn-seconds)
      FRESHNESS_WARN_SECONDS="${2:-$FRESHNESS_WARN_SECONDS}"
      shift 2
      ;;
    --no-discovery)
      DISCOVERY=0
      shift
      ;;
    --help|-h)
      printf '%s\n' "Usage: sh scripts/check_api.sh [--base-url http://127.0.0.1:8000] [--timeout 1.5] [--freshness-warn-seconds 120] [--no-discovery]"
      exit 0
      ;;
    *)
      add_issue "Unknown argument: $1"
      shift
      ;;
  esac
done

if ! command -v curl >/dev/null 2>&1; then
  add_issue "curl not found"
  add_error "curl command is required for API checks"
  printf '{"status":"critical","issues":[%s],"metrics":{"api_status":"down","latency_ms":null,"model_loaded":false,"data_feed_age":null,"errors":[%s]}}\n' "${ISSUES_ITEMS#,}" "${ERRORS_ITEMS#,}"
  exit 2
fi

HTTP_BODY=""
HTTP_CODE=""
HTTP_TIME=""

request_url() {
  req_url="$1"
  tmp_file=$(mktemp 2>/dev/null || printf '/tmp/check_api_%s.tmp' "$$")
  meta=$(curl -sS --max-time "$TIMEOUT_SECONDS" -o "$tmp_file" -w '%{http_code}|%{time_total}' "$req_url" 2>/dev/null)
  rc=$?
  if [ "$rc" -ne 0 ]; then
    rm -f "$tmp_file" >/dev/null 2>&1 || true
    return 1
  fi
  HTTP_CODE=$(printf '%s' "$meta" | cut -d'|' -f1)
  HTTP_TIME=$(printf '%s' "$meta" | cut -d'|' -f2)
  HTTP_BODY=$(cat "$tmp_file" 2>/dev/null || true)
  rm -f "$tmp_file" >/dev/null 2>&1 || true
  return 0
}

is_2xx() {
  code="$1"
  case "$code" in
    2??) return 0 ;;
    *) return 1 ;;
  esac
}

extract_json_number() {
  key="$1"
  body="$2"
  printf '%s' "$body" | tr -d '\n' | sed -nE "s/.*\"$key\"[[:space:]]*:[[:space:]]*(-?[0-9]+([.][0-9]+)?).*/\1/p" | head -n 1
}

json_has_true() {
  key="$1"
  body="$2"
  printf '%s' "$body" | tr -d '\n' | grep -E "\"$key\"[[:space:]]*:[[:space:]]*true" >/dev/null 2>&1
}

json_has_null() {
  key="$1"
  body="$2"
  printf '%s' "$body" | tr -d '\n' | grep -E "\"$key\"[[:space:]]*:[[:space:]]*null" >/dev/null 2>&1
}

json_extract_string() {
  key="$1"
  body="$2"
  printf '%s' "$body" | tr -d '\n' | sed -nE "s/.*\"$key\"[[:space:]]*:[[:space:]]*\"([^\"]*)\".*/\1/p" | head -n 1
}

STATUS_ENDPOINT=""
STATUS_BODY=""
LATENCY_MS="null"
selected_base="$BASE_URL"

try_base() {
  base="$1"
  for endpoint in /status /api/status; do
    if request_url "${base}${endpoint}"; then
      if is_2xx "$HTTP_CODE"; then
        STATUS_ENDPOINT="$endpoint"
        STATUS_BODY="$HTTP_BODY"
        LATENCY_MS=$(awk "BEGIN { printf \"%.2f\", ($HTTP_TIME + 0) * 1000 }")
        selected_base="$base"
        return 0
      fi
    fi
  done
  return 1
}

if ! try_base "$BASE_URL"; then
  if [ "$DISCOVERY" -eq 1 ]; then
    for port in 8000 8001 8002 8003 8004; do
      candidate="http://127.0.0.1:${port}"
      if try_base "$candidate"; then
        break
      fi
    done
  fi
fi

model_loaded=false
db_connected=false
data_feed_age="null"
portfolio_ok=false
trades_ok=false
model_endpoint_used=""

if [ -n "$STATUS_ENDPOINT" ]; then
  age_val=$(extract_json_number "data_age_seconds" "$STATUS_BODY")
  if [ -n "$age_val" ]; then
    data_feed_age="$age_val"
  fi

  if json_has_true "db_storage_ok" "$STATUS_BODY"; then
    db_connected=true
  else
    db_backend=$(json_extract_string "db_backend" "$STATUS_BODY")
    if [ -n "$db_backend" ] && [ "$db_backend" != "unknown" ]; then
      db_connected=true
    fi
  fi

  model_body=""
  for endpoint in /api/model_status /api/model-info /api/engine-state; do
    if request_url "${selected_base}${endpoint}"; then
      if is_2xx "$HTTP_CODE"; then
        model_body="$HTTP_BODY"
        model_endpoint_used="$endpoint"
        break
      fi
    fi
  done

  if [ -n "$model_body" ]; then
    if [ "$model_endpoint_used" = "/api/model-info" ]; then
      if ! json_has_null "active_model" "$model_body"; then
        model_loaded=true
      fi
    elif [ "$model_endpoint_used" = "/api/engine-state" ]; then
      if json_has_true "tft_disabled" "$model_body"; then
        model_loaded=false
      else
        active_model_name=$(json_extract_string "active_model_name" "$model_body")
        case "$active_model_name" in
          ""|xgb_meta_fallback|tft_disabled) model_loaded=false ;;
          *) model_loaded=true ;;
        esac
      fi
    else
      if json_has_true "model_loaded" "$model_body"; then
        model_loaded=true
      fi
    fi
  else
    add_issue "model status endpoint unavailable"
  fi

  for endpoint in /api/portfolio /api/portfolio/aggregate /api/metrics/portfolio; do
    if request_url "${selected_base}${endpoint}"; then
      if is_2xx "$HTTP_CODE"; then
        portfolio_ok=true
        break
      fi
    fi
  done
  if [ "$portfolio_ok" != "true" ]; then
    add_issue "portfolio endpoint unavailable"
  fi

  for endpoint in /api/trades /trades; do
    if request_url "${selected_base}${endpoint}"; then
      if is_2xx "$HTTP_CODE"; then
        trades_ok=true
        break
      fi
    fi
  done
  if [ "$trades_ok" != "true" ]; then
    add_issue "trades endpoint unavailable"
  fi
else
  add_issue "API status endpoint unreachable"
  add_error "Unable to reach /status or /api/status on configured/discovered hosts"
fi

if [ "$db_connected" != "true" ]; then
  add_issue "database connectivity flag is not healthy"
fi
if [ "$model_loaded" != "true" ]; then
  add_issue "model not loaded or fallback mode is active"
fi
if [ "$data_feed_age" != "null" ]; then
  is_stale=$(awk "BEGIN {print (($data_feed_age + 0) > ($FRESHNESS_WARN_SECONDS + 0)) ? 1 : 0}")
  if [ "$is_stale" -eq 1 ]; then
    add_issue "market data appears stale (data_feed_age=${data_feed_age}s)"
  fi
else
  add_issue "market data freshness unavailable"
fi

api_status="up"
status="ok"
exit_code=0

if [ -z "$STATUS_ENDPOINT" ]; then
  api_status="down"
  status="critical"
  exit_code=2
elif [ -n "${ISSUES_ITEMS#,}" ]; then
  api_status="degraded"
  status="warning"
  exit_code=1
fi

printf '{'
printf '"status":"%s",' "$status"
printf '"issues":[%s],' "${ISSUES_ITEMS#,}"
printf '"metrics":{'
printf '"base_url":"%s",' "$(json_escape "$selected_base")"
printf '"status_endpoint":"%s",' "$(json_escape "$STATUS_ENDPOINT")"
printf '"api_status":"%s",' "$api_status"
printf '"latency_ms":%s,' "$LATENCY_MS"
printf '"model_loaded":%s,' "$model_loaded"
printf '"data_feed_age":%s,' "$data_feed_age"
printf '"db_connected":%s,' "$db_connected"
printf '"portfolio_endpoint_ok":%s,' "$portfolio_ok"
printf '"trades_endpoint_ok":%s,' "$trades_ok"
printf '"errors":[%s]' "${ERRORS_ITEMS#,}"
printf '}'
printf '}\n'

exit "$exit_code"
