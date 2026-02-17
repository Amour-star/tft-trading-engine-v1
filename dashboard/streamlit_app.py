"""
Streamlit admin dashboard for the TFT Trading Engine.
"""
from __future__ import annotations

import os
import requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

API_BASE = os.getenv("API_BASE", "http://localhost:8000/api")

st.set_page_config(page_title="TFT Trading Engine", page_icon="📈", layout="wide")


def api_get(endpoint, params=None):
    try:
        r = requests.get(f"{API_BASE}/{endpoint}", params=params, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.error(f"API Error: {exc}")
        return None


def api_post(endpoint, data=None):
    try:
        r = requests.post(f"{API_BASE}/{endpoint}", json=data, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.error(f"API Error: {exc}")
        return None


status = api_get("status")

# -- Sidebar Controls --
with st.sidebar:
    st.title("Engine Controls")

    if status:
        mode = status.get("mode", "UNKNOWN")
        mode_badge = "🧪" if mode == "PAPER" else "🔴"
        st.metric("Mode", f"{mode} {mode_badge}")
        st.metric("Engine", "Running" if status.get("engine_running") else "Stopped")
        st.metric("Daily PnL", f"${status.get('daily_pnl', 0):.2f}")

        if mode == "PAPER":
            st.metric("Virtual Balance", f"${status.get('virtual_balance', 0):.2f}")
            st.metric("Paper Realized", f"${status.get('paper_realized_pnl', 0):.2f}")
            st.metric("Paper Unrealized", f"${status.get('paper_unrealized_pnl', 0):.2f}")
        elif status.get("balance") is not None:
            st.metric("Live Balance", f"${status.get('balance', 0):.2f}")

        if status.get("paused"):
            st.warning("Trading PAUSED")
        if status.get("killed"):
            st.error("KILL SWITCH ACTIVE")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Pause", use_container_width=True):
            api_post("control", {"action": "pause"})
            st.rerun()
    with c2:
        if st.button("Resume", use_container_width=True):
            api_post("control", {"action": "resume"})
            st.rerun()

    st.divider()
    confirm_kill = st.checkbox("Confirm kill")
    if st.button("EMERGENCY KILL", type="primary", use_container_width=True):
        if confirm_kill:
            api_post("control", {"action": "kill"})
            st.rerun()

    if st.button("Reset Kill", use_container_width=True):
        api_post("control", {"action": "reset_kill"})
        st.rerun()

    st.divider()
    st.subheader("Thresholds")
    thresholds = api_get("thresholds") or {}
    new_conf = st.slider("Confidence", 0.50, 0.95, float(thresholds.get("confidence_threshold", 0.70)), 0.05)
    new_risk = st.slider("Risk %", 0.5, 3.0, float(thresholds.get("risk_per_trade", 0.01)) * 100, 0.25)
    if st.button("Update Thresholds"):
        api_post("thresholds", {"confidence_threshold": new_conf, "risk_per_trade": new_risk / 100})
        st.success("Updated")


# -- Main --
st.title("TFT AI Trading Engine")

stats = api_get("stats") or {}
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Trades", stats.get("total_trades", 0))
c2.metric("Win Rate", f"{stats.get('win_rate', 0):.1%}")
c3.metric("Total PnL", f"${stats.get('total_pnl', 0):.2f}")
c4.metric("Avg R", f"{stats.get('avg_r_multiple', 0):.2f}R")
c5.metric("W/L", f"{stats.get('winning_trades', 0)}/{stats.get('losing_trades', 0)}")

if status and status.get("mode") == "PAPER":
    realized = float(status.get("paper_realized_pnl", 0) or 0)
    unrealized = float(status.get("paper_unrealized_pnl", 0) or 0)
    virtual_balance = float(status.get("virtual_balance", 0) or 0)
    p1, p2, p3 = st.columns(3)
    p1.metric("Paper Balance", f"${virtual_balance:.2f}")
    p2.metric("Realized PnL", f"${realized:.2f}")
    p3.metric("Unrealized PnL", f"${unrealized:.2f}")


TAB_NAMES = ["📈 PnL Curve", "📋 Trades", "🔮 Predictions", "🧠 Learning", "📊 Open Trade"]
tab1, tab2, tab3, tab4, tab5 = st.tabs(TAB_NAMES)

with tab1:
    curve = api_get("pnl-curve", {"days": 30})
    if curve:
        df = pd.DataFrame(curve)
        if not df.empty:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=df["time"],
                    y=df["pnl"],
                    mode="lines+markers",
                    name="Cumulative PnL",
                    line=dict(color="cyan", width=2),
                )
            )
            fig.update_layout(
                title="Cumulative PnL",
                xaxis_title="Time",
                yaxis_title="PnL ($)",
                template="plotly_dark",
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No closed trades yet")
    else:
        st.info("No data available")

with tab2:
    trades = api_get("trades", {"limit": 50})
    if trades:
        df = pd.DataFrame(trades)
        if not df.empty:
            st.dataframe(
                df[
                    [
                        "trade_id",
                        "pair",
                        "side",
                        "entry_time",
                        "exit_time",
                        "entry_price",
                        "exit_price",
                        "pnl",
                        "pnl_pct",
                        "r_multiple",
                        "exit_reason",
                        "confidence",
                        "status",
                    ]
                ],
                use_container_width=True,
                height=500,
            )

            for _, row in df.iterrows():
                if row.get("ai_reasoning"):
                    with st.expander(f"AI Reasoning: {row['trade_id']}"):
                        st.text(row["ai_reasoning"])
    else:
        st.info("No trades yet")

with tab3:
    preds = api_get("predictions", {"limit": 10})
    if preds:
        df = pd.DataFrame(preds)
        if not df.empty:
            st.dataframe(df, use_container_width=True)

            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=df["pair"],
                    y=df["confidence"],
                    name="Confidence",
                    marker_color=[
                        "green" if c > 0.7 else "orange" if c > 0.5 else "red"
                        for c in df["confidence"]
                    ],
                )
            )
            fig.update_layout(title="Last 10 Prediction Confidences", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No predictions yet")

with tab4:
    learning = api_get("learning-metrics", {"limit": 20})
    if learning:
        df = pd.DataFrame(learning)
        if not df.empty:
            st.dataframe(df, use_container_width=True)

            st.subheader("Pattern Analysis")
            total = len(df)
            vol_misread = df["volatility_misread"].sum()
            conf_overest = df["confidence_overestimated"].sum()
            stop_tight = df["stop_too_tight"].sum()
            c1, c2, c3 = st.columns(3)
            c1.metric("Volatility Misreads", f"{vol_misread}/{total}")
            c2.metric("Confidence Overest.", f"{conf_overest}/{total}")
            c3.metric("Stops Too Tight", f"{stop_tight}/{total}")
    else:
        st.info("No learning data yet")

with tab5:
    if status and status.get("mode") == "PAPER":
        paper_positions = status.get("paper_positions") or []
        if paper_positions:
            st.subheader("Open Paper Positions")
            st.dataframe(pd.DataFrame(paper_positions), use_container_width=True)
        else:
            st.info("No open paper positions")

    if status and status.get("open_trade"):
        trade = status["open_trade"]
        st.subheader(f"Open: {trade['pair']}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Entry", f"${trade['entry_price']:.6f}")
        c2.metric("Stop", f"${trade['stop_price']:.6f}")
        c3.metric("Target", f"${trade['target_price']:.6f}")
        c4.metric("Confidence", f"{trade['confidence']:.3f}")

        if st.button("Force Close Position", type="primary"):
            api_post("control", {"action": "force_close"})
            st.rerun()
    else:
        st.info("No open position")

model = api_get("model-info")
if model and model.get("active_model"):
    with st.expander("Active Model"):
        st.json(model["active_model"])

st.markdown("---")
st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
if st.checkbox("Auto-refresh (10s)"):
    import time

    time.sleep(10)
    st.rerun()
