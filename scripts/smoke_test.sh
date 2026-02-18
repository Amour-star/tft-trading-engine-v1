#!/usr/bin/env bash
# =============================================================================
# DELIVERABLE 4 — Post-deployment smoke test checklist
#
# Run from the project root on the Oracle Cloud VM:
#   bash scripts/smoke_test.sh
#
# Optionally pass the public IP as $1 (auto-detected if omitted):
#   bash scripts/smoke_test.sh 92.5.10.230
# =============================================================================
set -uo pipefail

PUBLIC_IP="${1:-$(curl -s --max-time 5 ifconfig.me 2>/dev/null || echo '92.5.10.230')}"
PASS=0
FAIL=0

green()  { printf "\033[32m%s\033[0m\n" "$*"; }
red()    { printf "\033[31m%s\033[0m\n" "$*"; }
header() { printf "\n\033[1;36m== %s ==\033[0m\n" "$*"; }

check() {
    local desc="$1"; shift
    if "$@" >/dev/null 2>&1; then
        green "[PASS] $desc"
        ((PASS++))
    else
        red  "[FAIL] $desc"
        ((FAIL++))
    fi
}

# =========================================================================
header "1. Port 8501 open at OS level"
# =========================================================================

check "iptables allows port 8501" \
    bash -c "sudo iptables -L INPUT -n 2>/dev/null | grep -q 8501 \
             || sudo firewall-cmd --list-ports 2>/dev/null | grep -q 8501 \
             || sudo ufw status 2>/dev/null | grep -q 8501"

check "Something is listening on port 8501" \
    bash -c "ss -tlnp | grep -q ':8501'"

# =========================================================================
header "2. Dashboard accessible externally"
# =========================================================================

check "Streamlit health endpoint (localhost)" \
    curl -sf --max-time 5 http://localhost:8501/_stcore/health

check "Streamlit health endpoint (public IP)" \
    curl -sf --max-time 5 "http://${PUBLIC_IP}:8501/_stcore/health"

check "Dashboard API /api/status (localhost)" \
    curl -sf --max-time 5 http://localhost:8000/api/status

# =========================================================================
header "3. Containers running without GPU"
# =========================================================================

check "docker compose services are up" \
    docker compose ps --status running | grep -q "engine"

check "No nvidia/GPU runtime in running containers" \
    bash -c "! docker inspect \$(docker compose ps -q engine) 2>/dev/null \
        | grep -qi 'nvidia'"

check "CUDA_VISIBLE_DEVICES is empty inside engine container" \
    bash -c "docker compose exec -T engine printenv CUDA_VISIBLE_DEVICES | grep -q '^$'"

check "FORCE_CPU=1 is set inside engine container" \
    bash -c "docker compose exec -T engine printenv FORCE_CPU | grep -q '1'"

check "PyTorch sees no CUDA (GPU available: False)" \
    bash -c "docker compose exec -T engine python -c \
        'import torch; assert not torch.cuda.is_available(), \"GPU should be disabled\"'"

# =========================================================================
header "4. Engine inference cycle working on CPU"
# =========================================================================

check "Engine container is running and healthy" \
    bash -c "docker compose ps engine | grep -qE 'running|Up'"

check "Engine logs show CPU mode (no CUDA)" \
    bash -c "docker compose logs --tail=100 engine 2>&1 | grep -iqE 'cpu|cuda.*false|no gpu|accelerator.*cpu'"

check "Engine logs show recent activity (last 5 min)" \
    bash -c "docker compose logs --tail=50 --since=5m engine 2>&1 | grep -qE '.'"

check "No CUDA/GPU errors in logs" \
    bash -c "! docker compose logs --tail=200 engine 2>&1 | grep -iqE 'cuda error|gpu error|RuntimeError.*cuda'"

# =========================================================================
header "Summary"
# =========================================================================

TOTAL=$((PASS + FAIL))
echo ""
echo "Passed: $PASS / $TOTAL"
if [ "$FAIL" -gt 0 ]; then
    red "Failed: $FAIL / $TOTAL"
    echo ""
    red "Some checks failed. Review the output above."
    exit 1
else
    green "All checks passed!"
    exit 0
fi
