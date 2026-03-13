#!/usr/bin/env sh
# Usage:
#   sh scripts/check_docker.sh [--compose-file docker-compose.yml] [--log-tail 120]
#                              [--restart-threshold 3] [--memory-warn-mb 4096]
#                              [--max-log-containers 4]
#
# Purpose:
#   Read-only Docker health diagnostics for automation.
#   This script never modifies containers or trading state.

set -u

COMPOSE_FILE="docker-compose.yml"
LOG_TAIL=120
RESTART_THRESHOLD=3
MEMORY_WARN_MB=4096
MAX_LOG_CONTAINERS=4
ERROR_PATTERN='error|traceback|critical|fatal|panic|failed|exception'

json_escape() {
  printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g; s/\r//g; :a;N;$!ba;s/\n/\\n/g'
}

count_non_empty_lines() {
  printf '%s\n' "$1" | sed '/^[[:space:]]*$/d' | wc -l | tr -d ' '
}

mem_to_mb() {
  value="$1"
  number=$(printf '%s' "$value" | sed -E 's/^([0-9]+([.][0-9]+)?).*/\1/')
  unit=$(printf '%s' "$value" | sed -E 's/^[0-9]+([.][0-9]+)?([A-Za-z]+).*/\2/')
  case "$unit" in
    B) factor=0.000001 ;;
    kB|KB|KiB) factor=0.001 ;;
    MB|MiB) factor=1 ;;
    GB|GiB) factor=1024 ;;
    TB|TiB) factor=1048576 ;;
    *) factor=0 ;;
  esac
  awk "BEGIN { printf \"%.2f\", ($number + 0) * ($factor + 0) }"
}

ISSUES_ITEMS=""
RECENT_ERRORS_ITEMS=""
UNHEALTHY_ITEMS=""

add_issue() {
  esc=$(json_escape "$1")
  ISSUES_ITEMS="$ISSUES_ITEMS,\"$esc\""
}

add_recent_error() {
  esc=$(json_escape "$1")
  RECENT_ERRORS_ITEMS="$RECENT_ERRORS_ITEMS,\"$esc\""
}

add_unhealthy() {
  esc=$(json_escape "$1")
  UNHEALTHY_ITEMS="$UNHEALTHY_ITEMS,\"$esc\""
}

while [ $# -gt 0 ]; do
  case "$1" in
    --compose-file)
      COMPOSE_FILE="${2:-$COMPOSE_FILE}"
      shift 2
      ;;
    --log-tail)
      LOG_TAIL="${2:-$LOG_TAIL}"
      shift 2
      ;;
    --restart-threshold)
      RESTART_THRESHOLD="${2:-$RESTART_THRESHOLD}"
      shift 2
      ;;
    --memory-warn-mb)
      MEMORY_WARN_MB="${2:-$MEMORY_WARN_MB}"
      shift 2
      ;;
    --max-log-containers)
      MAX_LOG_CONTAINERS="${2:-$MAX_LOG_CONTAINERS}"
      shift 2
      ;;
    --help|-h)
      printf '%s\n' "Usage: sh scripts/check_docker.sh [--compose-file docker-compose.yml] [--log-tail 120] [--restart-threshold 3] [--memory-warn-mb 4096] [--max-log-containers 4]"
      exit 0
      ;;
    *)
      add_issue "Unknown argument: $1"
      shift
      ;;
  esac
done

if ! command -v docker >/dev/null 2>&1; then
  add_issue "docker CLI not found"
  printf '{"status":"critical","issues":[%s],"metrics":{"containers_running":0,"unhealthy_containers":[],"restart_count":0,"recent_errors":[],"memory_usage":{"total_mb":0,"per_container":{}}}}\n' "${ISSUES_ITEMS#,}"
  exit 2
fi

if ! docker info >/dev/null 2>&1; then
  add_issue "docker daemon unavailable"
  printf '{"status":"critical","issues":[%s],"metrics":{"containers_running":0,"unhealthy_containers":[],"restart_count":0,"recent_errors":[],"memory_usage":{"total_mb":0,"per_container":{}}}}\n' "${ISSUES_ITEMS#,}"
  exit 2
fi

HAS_COMPOSE=0
if docker compose version >/dev/null 2>&1; then
  HAS_COMPOSE=1
fi

RUNNING_IDS=""
ALL_IDS=""
SCOPE="docker"
if [ "$HAS_COMPOSE" -eq 1 ] && [ -f "$COMPOSE_FILE" ]; then
  SCOPE="docker_compose"
  RUNNING_IDS=$(docker compose -f "$COMPOSE_FILE" ps -q 2>/dev/null || true)
  ALL_IDS=$(docker compose -f "$COMPOSE_FILE" ps -aq 2>/dev/null || true)
  if [ -z "$(printf '%s' "$ALL_IDS" | sed '/^[[:space:]]*$/d')" ]; then
    # Fallback when compose project name/context does not match running stack.
    SCOPE="docker"
    RUNNING_IDS=$(docker ps -q 2>/dev/null || true)
    ALL_IDS="$RUNNING_IDS"
  fi
else
  RUNNING_IDS=$(docker ps -q 2>/dev/null || true)
  ALL_IDS="$RUNNING_IDS"
fi

containers_running=$(count_non_empty_lines "$RUNNING_IDS")
containers_total=$(count_non_empty_lines "$ALL_IDS")
if [ "$containers_total" -gt 0 ] && [ "$containers_running" -eq 0 ]; then
  add_issue "no running containers detected in scope=${SCOPE}"
fi

restart_count_total=0
critical_flag=0

MEMORY_ITEMS=""
TOTAL_MEMORY_MB="0.00"

if [ "$containers_running" -gt 0 ]; then
  for cid in $RUNNING_IDS; do
    cname=$(docker inspect --format '{{.Name}}' "$cid" 2>/dev/null | sed 's#^/##')
    if [ -z "$cname" ]; then
      cname="$cid"
    fi

    health=$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "$cid" 2>/dev/null || printf 'unknown')
    restart_count=$(docker inspect --format '{{.RestartCount}}' "$cid" 2>/dev/null || printf '0')
    case "$restart_count" in
      ''|*[!0-9]*) restart_count=0 ;;
    esac

    restart_count_total=$(expr "$restart_count_total" + "$restart_count")
    if [ "$restart_count" -ge "$RESTART_THRESHOLD" ]; then
      add_issue "container ${cname} restart_count=${restart_count} (threshold=${RESTART_THRESHOLD})"
    fi
    high_restart_threshold=$(expr "$RESTART_THRESHOLD" \* 2)
    if [ "$restart_count" -ge "$high_restart_threshold" ] && [ "$high_restart_threshold" -gt 0 ]; then
      critical_flag=1
    fi

    if [ "$health" = "unhealthy" ] || [ "$health" = "starting" ] || [ "$health" = "unknown" ]; then
      add_unhealthy "$cname:$health"
      add_issue "container ${cname} health=${health}"
      if [ "$health" = "unhealthy" ]; then
        critical_flag=1
      fi
    fi
  done

  stats_lines=$(docker stats --no-stream --format '{{.Container}}|{{.Name}}|{{.MemUsage}}' $RUNNING_IDS 2>/dev/null || true)
  if [ -n "$stats_lines" ]; then
    printf '%s\n' "$stats_lines" | while IFS='|' read -r sid sname mem_usage; do
      [ -z "$sname" ] && continue
      used=$(printf '%s' "$mem_usage" | sed 's#/.*##' | tr -d ' ')
      used_mb=$(mem_to_mb "$used")

      over_warn=$(awk "BEGIN {print (($used_mb + 0) > ($MEMORY_WARN_MB + 0)) ? 1 : 0}")
      if [ "$over_warn" -eq 1 ]; then
        printf '%s\n' "WARN_MEMORY|$sname|$used_mb"
      fi
      printf '%s\n' "MEM|$sname|$mem_usage|$used_mb"
    done > /tmp/check_docker_mem.$$ 2>/dev/null

    if [ -f /tmp/check_docker_mem.$$ ]; then
      while IFS='|' read -r tag a b c; do
        if [ "$tag" = "WARN_MEMORY" ]; then
          add_issue "container ${a} memory_mb=${b} exceeds threshold=${MEMORY_WARN_MB}"
          continue
        fi
        if [ "$tag" = "MEM" ]; then
          esc_name=$(json_escape "$a")
          esc_raw=$(json_escape "$b")
          used_mb="$c"
          MEMORY_ITEMS="$MEMORY_ITEMS,\"$esc_name\":{\"raw\":\"$esc_raw\",\"used_mb\":$used_mb}"
          TOTAL_MEMORY_MB=$(awk "BEGIN {printf \"%.2f\", ($TOTAL_MEMORY_MB + 0) + ($used_mb + 0)}")
        fi
      done < /tmp/check_docker_mem.$$
      rm -f /tmp/check_docker_mem.$$
    fi
  fi

  scanned=0
  for cid in $RUNNING_IDS; do
    if [ "$scanned" -ge "$MAX_LOG_CONTAINERS" ]; then
      break
    fi
    scanned=$(expr "$scanned" + 1)
    cname=$(docker inspect --format '{{.Name}}' "$cid" 2>/dev/null | sed 's#^/##')
    [ -z "$cname" ] && cname="$cid"
    matched=$(
      docker logs --tail "$LOG_TAIL" "$cid" 2>&1 \
        | grep -Ei "$ERROR_PATTERN" \
        | grep -Evi '"exception"[[:space:]]*:[[:space:]]*null' \
        | tail -n 2 \
        || true
    )
    if [ -n "$matched" ]; then
      while IFS= read -r line; do
        [ -z "$line" ] && continue
        raw="${cname}: ${line}"
        raw_len=$(printf '%s' "$raw" | wc -c | tr -d ' ')
        if [ "$raw_len" -gt 260 ]; then
          raw="$(printf '%s' "$raw" | cut -c1-260)..."
        fi
        add_recent_error "$raw"
      done <<EOF
$matched
EOF
    fi
  done
fi

if [ -n "${RECENT_ERRORS_ITEMS#,}" ]; then
  add_issue "recent error patterns detected in container logs"
fi

if [ "$containers_total" -gt "$containers_running" ]; then
  add_issue "some compose containers are not running (${containers_running}/${containers_total})"
fi

if [ -n "${ISSUES_ITEMS#,}" ]; then
  status="warning"
  exit_code=1
else
  status="ok"
  exit_code=0
fi

if [ "$critical_flag" -eq 1 ]; then
  status="critical"
  exit_code=2
fi

printf '{'
printf '"status":"%s",' "$status"
printf '"issues":[%s],' "${ISSUES_ITEMS#,}"
printf '"metrics":{'
printf '"scope":"%s",' "$SCOPE"
printf '"containers_running":%s,' "$containers_running"
printf '"containers_total":%s,' "$containers_total"
printf '"unhealthy_containers":[%s],' "${UNHEALTHY_ITEMS#,}"
printf '"restart_count":%s,' "$restart_count_total"
printf '"recent_errors":[%s],' "${RECENT_ERRORS_ITEMS#,}"
printf '"memory_usage":{"total_mb":%s,"per_container":{%s}}' "$TOTAL_MEMORY_MB" "${MEMORY_ITEMS#,}"
printf '}'
printf '}\n'

exit "$exit_code"
