"""
Streamlit dashboard for TFT Trading Engine v2.
"""
from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_BASE = os.getenv("TFT_API_BASE", "http://localhost:8000/api")

st.set_page_config(page_title="TFT Engine v2", page_icon="TFT", layout="wide")


def api_get(endpoint: str, params: dict | None = None):
    try:
        r = requests.get(f"{API_BASE}/{endpoint}", params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.error(f"API error: {exc}")
        return None


def api_post(endpoint: str, payload: dict | None = None):
    try:
        r = requests.post(f"{API_BASE}/{endpoint}", json=payload or {}, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.error(f"API error: {exc}")
        return None


st.title("TFT Trading Engine v2 Dashboard")

score = api_get("ai/current-score") or {}
history = api_get("ai/history", {"limit": 180}) or []
perf = api_get("metrics/latest") or {}
open_trades = api_get("trades/open") or []

top1, top2, top3 = st.columns(3)
top1.metric("AI Confidence", f"{float(score.get('confidence', 0.0)):.2%}")
top2.metric("Current Win Rate", f"{float(score.get('win_rate', 0.0)):.2%}")
top3.metric("Model Version", score.get("model_version", "unknown"))

st.subheader("Trade Confidence Visualization")
confidence = float(score.get("confidence", 0.0))
gauge_color = "green" if confidence > 0.70 else "yellow" if confidence >= 0.50 else "red"
gauge = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": gauge_color},
            "steps": [
                {"range": [0, 50], "color": "#8b0000"},
                {"range": [50, 70], "color": "#c9a227"},
                {"range": [70, 100], "color": "#1f7a1f"},
            ],
        },
        title={"text": "AI Confidence"},
    )
)
st.plotly_chart(gauge, use_container_width=True)

st.subheader("Open Trades")
if open_trades:
    df_open = pd.DataFrame(open_trades)
    show_cols = ["id", "symbol", "side", "entry_price", "quantity", "cost_with_fees", "ai_confidence", "model_version"]
    st.dataframe(df_open[show_cols], use_container_width=True)
    for trade in open_trades:
        c1, c2 = st.columns([4, 1])
        with c1:
            st.caption(
                f"Trade #{trade['id']} | {trade['symbol']} | Qty {trade['quantity']:.6f} | "
                f"Cost+Fees {trade['cost_with_fees']:.4f}"
            )
        with c2:
            if st.button("Sell to USDC", key=f"sell_{trade['id']}", use_container_width=True):
                close_price = float(trade["entry_price"])
                api_post(f"trades/{trade['id']}/close", {"exit_price": close_price, "fees": 0.0})
                st.rerun()
else:
    st.info("No open trades.")

st.subheader("AI Score Chart Over Time")
if history:
    df_hist = pd.DataFrame(history)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_hist["date"], y=df_hist["win_rate"], mode="lines", name="Daily Win Rate"))
    fig.add_trace(go.Scatter(x=df_hist["date"], y=df_hist["avg_confidence"], mode="lines", name="Avg Confidence"))
    fig.add_trace(go.Scatter(x=df_hist["date"], y=df_hist["cumulative_return"], mode="lines", name="Cumulative Return"))
    fig.update_layout(height=420, xaxis_title="Date")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("AI history not available yet.")

st.subheader("Model Performance Panel")
p1, p2, p3, p4, p5 = st.columns(5)
p1.metric("Accuracy", f"{float(perf.get('accuracy', 0.0)):.2%}")
p2.metric("Sharpe Ratio", f"{float(perf.get('sharpe_ratio', 0.0)):.3f}")
p3.metric("Max Drawdown", f"{float(perf.get('max_drawdown', 0.0)):.2%}")
p4.metric("Sortino Ratio", f"{float(perf.get('sortino_ratio', 0.0)):.3f}")
p5.metric("Profit Factor", f"{float(perf.get('profit_factor', 0.0)):.3f}")

st.caption(f"Updated at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")

