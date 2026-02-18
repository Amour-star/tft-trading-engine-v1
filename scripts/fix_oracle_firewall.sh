#!/usr/bin/env bash
# =============================================================================
# DELIVERABLE 1 — Oracle Cloud VM firewall fix for port 8501 (Streamlit)
#
# Run as root on the Oracle Cloud instance:
#   sudo bash scripts/fix_oracle_firewall.sh
#
# Covers both Oracle Linux (firewalld) and Ubuntu (iptables / ufw).
# After running this script you STILL need to open port 8501 in the
# Oracle Cloud Console Security List / NSG — see instructions at the end.
# =============================================================================
set -euo pipefail

PORT=8501

echo "===== Detecting OS ====="
if [ -f /etc/os-release ]; then
    . /etc/os-release
    echo "OS: $PRETTY_NAME"
else
    echo "Cannot detect OS — /etc/os-release not found."
    exit 1
fi

# -------------------------------------------------------------------------
# Oracle Linux / RHEL / CentOS — firewalld
# -------------------------------------------------------------------------
if command -v firewall-cmd &>/dev/null; then
    echo ""
    echo "===== firewalld detected — opening port $PORT/tcp ====="
    firewall-cmd --permanent --add-port=${PORT}/tcp
    firewall-cmd --reload
    echo "firewalld: port $PORT/tcp opened and persisted."

# -------------------------------------------------------------------------
# Ubuntu / Debian — iptables + iptables-persistent, or ufw
# -------------------------------------------------------------------------
elif command -v ufw &>/dev/null && ufw status | grep -q "Status: active"; then
    echo ""
    echo "===== ufw is active — opening port $PORT/tcp ====="
    ufw allow ${PORT}/tcp comment "Streamlit dashboard"
    ufw reload
    echo "ufw: port $PORT/tcp allowed."

else
    echo ""
    echo "===== Using raw iptables — opening port $PORT/tcp ====="

    # Insert at the top of the INPUT chain so it takes effect before any DROP
    iptables -I INPUT 1 -p tcp --dport ${PORT} -j ACCEPT

    # Persist across reboots
    if command -v netfilter-persistent &>/dev/null; then
        netfilter-persistent save
        echo "iptables rule persisted via netfilter-persistent."
    elif [ -d /etc/iptables ]; then
        iptables-save > /etc/iptables/rules.v4
        echo "iptables rule saved to /etc/iptables/rules.v4."
    else
        echo ""
        echo "WARNING: iptables-persistent not installed."
        echo "Install it to survive reboots:"
        echo "  apt-get install -y iptables-persistent"
        echo "  netfilter-persistent save"
        echo ""
        # Save anyway if the directory exists after install
        mkdir -p /etc/iptables
        iptables-save > /etc/iptables/rules.v4
    fi
fi

# -------------------------------------------------------------------------
# Verify
# -------------------------------------------------------------------------
echo ""
echo "===== Verifying port $PORT is open at OS level ====="
if command -v ss &>/dev/null; then
    ss -tlnp | grep ":${PORT}" || echo "(No process listening on $PORT yet — start the container first)"
elif command -v netstat &>/dev/null; then
    netstat -tlnp | grep ":${PORT}" || echo "(No process listening on $PORT yet — start the container first)"
fi

echo ""
echo "===== OS-level firewall fix complete ====="
echo ""
echo "IMPORTANT: You must ALSO open port $PORT in Oracle Cloud Console:"
echo "  1. Go to: Networking > Virtual Cloud Networks > your VCN"
echo "  2. Click the subnet used by your instance"
echo "  3. Click the Security List (e.g. 'Default Security List')"
echo "  4. Add Ingress Rule:"
echo "       Source CIDR : 0.0.0.0/0"
echo "       IP Protocol : TCP"
echo "       Dest Port   : $PORT"
echo "  5. Save. Takes effect within ~30 seconds."
echo ""
echo "Then test: curl -s -o /dev/null -w '%{http_code}' http://$(curl -s ifconfig.me):${PORT}/_stcore/health"
