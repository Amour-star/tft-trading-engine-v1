"""
Streamlit admin dashboard for the TFT Trading Engine.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path
import requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

from config.settings import settings

PAIRS = ["BTC-USDT", "ETH-USDT", "XRP-USDT", "DOGE-USDT"]
API_BASE_DEFAULT = os.getenv("API_BASE", "http://localhost:8000/api")
API_BASE_BY_SYMBOL = {
    "BTC-USDT": os.getenv("API_BASE_BTC", API_BASE_DEFAULT),
    "ETH-USDT": os.getenv("API_BASE_ETH", API_BASE_DEFAULT),
    "XRP-USDT": os.getenv("API_BASE_XRP", API_BASE_DEFAULT),
    "DOGE-USDT": os.getenv("API_BASE_DOGE", API_BASE_DEFAULT),
}

st.set_page_config(page_title="TFT Trading Engine", page_icon=":chart_with_upwards_trend:", layout="wide")


if "symbol" not in st.session_state:
    st.session_state.symbol = "XRP-USDT"

selected_pair = st.sidebar.selectbox(
    "Select Trading Pair",
    PAIRS,
    index=PAIRS.index(st.session_state.symbol) if st.session_state.symbol in PAIRS else 0,
)

if st.session_state.symbol != selected_pair:
    st.session_state.symbol = selected_pair
    st.rerun()

symbol = st.session_state.symbol


def get_api_base(selected_symbol: str) -> str:
    return API_BASE_BY_SYMBOL.get(selected_symbol, API_BASE_DEFAULT)


def get_db_path(selected_symbol: str) -> str:
    symbol_slug = selected_symbol.split("-")[0].lower()
    candidates = [
        Path(f"/app/state/{symbol_slug}/trades.db"),
        Path(f"state/{symbol_slug}/trades.db"),
        Path(f"/app/state/{symbol_slug}/tft_engine.db"),
        Path(f"state/{symbol_slug}/tft_engine.db"),
        Path(f"/app/state/{symbol_slug}/paper_trading.db"),
        Path(f"state/{symbol_slug}/paper_trading.db"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def compute_max_drawdown(equity_series: pd.Series) -> float:
    if equity_series.empty:
        return 0.0
    rolling_peak = equity_series.cummax()
    drawdown = (equity_series - rolling_peak) / rolling_peak.replace(0, pd.NA)
    return safe_float(drawdown.min(), 0.0)


def _safe_read_sql(selected_symbol: str, query: str, params: tuple | None = None) -> pd.DataFrame:
    db_path = get_db_path(selected_symbol)
    if not Path(db_path).exists():
        return pd.DataFrame()
    conn = sqlite3.connect(db_path)
    try:
        return pd.read_sql_query(query, conn, params=params or ())
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()


def load_trades(selected_symbol: str, limit: int = 50) -> pd.DataFrame:
    return _safe_read_sql(
        selected_symbol,
        "SELECT * FROM trades ORDER BY exit_time DESC LIMIT ?",
        (int(limit),),
    )


def load_pnl(selected_symbol: str) -> pd.DataFrame:
    return _safe_read_sql(
        selected_symbol,
        "SELECT exit_time AS time, pnl FROM trades WHERE exit_time IS NOT NULL ORDER BY exit_time ASC",
    )


def load_metrics(selected_symbol: str, limit: int = 300) -> pd.DataFrame:
    return _safe_read_sql(
        selected_symbol,
        "SELECT * FROM risk_metrics ORDER BY timestamp DESC LIMIT ?",
        (int(limit),),
    )


def load_ai_metrics(selected_symbol: str, limit: int = 180) -> pd.DataFrame:
    return _safe_read_sql(
        selected_symbol,
        "SELECT * FROM ai_score_history ORDER BY timestamp DESC LIMIT ?",
        (int(limit),),
    )


def load_open_positions(selected_symbol: str) -> pd.DataFrame:
    return _safe_read_sql(
        selected_symbol,
        "SELECT * FROM trades WHERE status IN ('OPEN', 'open') ORDER BY entry_time DESC",
    )


def load_symbol_snapshot(selected_symbol: str) -> dict:
    status_payload = api_get("status", selected_symbol=selected_symbol) or {}
    stats_payload = api_get("stats", selected_symbol=selected_symbol) or {}
    ai_payload = api_get("ai/current-score", selected_symbol=selected_symbol) or {}

    if (status_payload.get("mode") or "").upper() == "PAPER":
        balance = safe_float(status_payload.get("virtual_balance"))
    else:
        balance = safe_float(status_payload.get("balance"))

    realized = safe_float(status_payload.get("paper_realized_pnl"))
    unrealized = safe_float(status_payload.get("paper_unrealized_pnl"))
    win_rate = safe_float(stats_payload.get("win_rate"))
    open_positions = status_payload.get("paper_positions") or []
    exposure_value = sum(
        abs(safe_float(position.get("quantity")) * safe_float(position.get("mark_price")))
        for position in open_positions
    )
    regime = status_payload.get("latest_regime") or {}
    regime_label = (
        f"{regime.get('trend', 'n/a')} / {regime.get('volatility', 'n/a')}"
        if regime
        else "n/a"
    )

    return {
        "Symbol": selected_symbol,
        "Balance": balance,
        "Realized": realized,
        "Unrealized": unrealized,
        "WinRate": win_rate,
        "OpenTrades": int(len(open_positions)),
        "AIScore": safe_float(ai_payload.get("ai_score")),
        "AIConfidence": safe_float(ai_payload.get("confidence")),
        "Regime": regime_label,
        "ExposureValue": exposure_value,
    }


def build_portfolio_df(symbols: list[str]) -> pd.DataFrame:
    rows = [load_symbol_snapshot(sym) for sym in symbols]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "Symbol",
                "Balance",
                "Realized",
                "Unrealized",
                "WinRate",
                "OpenTrades",
                "Regime",
                "AIScore",
                "AIConfidence",
                "ExposureValue",
            ]
        )
    return df


def render_portfolio_overview(symbols: list[str]) -> None:
    st.subheader("Portfolio Overview")
    portfolio_df = build_portfolio_df(symbols)
    if portfolio_df.empty:
        st.info("No portfolio data available yet.")
        return

    total_equity = safe_float(portfolio_df["Balance"].sum())
    total_realized = safe_float(portfolio_df["Realized"].sum())
    total_unrealized = safe_float(portfolio_df["Unrealized"].sum())
    total_exposure = safe_float(portfolio_df["ExposureValue"].sum())
    initial_balance = safe_float(os.getenv("PAPER_INITIAL_BALANCE", settings.trading.paper_initial_balance))
    baseline = initial_balance * max(len(symbols), 1)
    portfolio_return = ((total_equity - baseline) / baseline) if baseline else 0.0

    pnl_history_frames = []
    for sym in symbols:
        df_pnl = load_pnl(sym)
        if df_pnl.empty:
            continue
        df_local = df_pnl.copy()
        df_local["time"] = pd.to_datetime(df_local["time"], errors="coerce")
        df_local["pnl"] = pd.to_numeric(df_local["pnl"], errors="coerce").fillna(0.0)
        pnl_history_frames.append(df_local[["time", "pnl"]])
    if pnl_history_frames:
        merged = pd.concat(pnl_history_frames, ignore_index=True).sort_values("time")
        merged["equity"] = baseline + merged["pnl"].cumsum()
        max_drawdown = compute_max_drawdown(merged["equity"])
    else:
        max_drawdown = 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Equity", f"${total_equity:,.2f}")
    m2.metric("Total Realized", f"${total_realized:,.2f}")
    m3.metric("Total Unrealized", f"${total_unrealized:,.2f}")
    m4.metric("Portfolio Return", f"{portfolio_return:.2%}")

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Portfolio Exposure", f"${total_exposure:,.2f}")
    m6.metric("Max Drawdown", f"{max_drawdown:.2%}")
    m7.metric("Open Trades", f"{int(portfolio_df['OpenTrades'].sum())}")
    m8.metric("Avg AI Confidence", f"{safe_float(portfolio_df['AIConfidence'].mean()):.2%}")

    pie_fig = go.Figure(
        data=[
            go.Pie(
                labels=portfolio_df["Symbol"],
                values=portfolio_df["Balance"],
                hole=0.4,
            )
        ]
    )
    pie_fig.update_layout(title="Asset Allocation by Balance", template="plotly_dark", height=320)
    st.plotly_chart(pie_fig, use_container_width=True)

    perf_df = portfolio_df[
        ["Symbol", "Balance", "Realized", "Unrealized", "WinRate", "OpenTrades", "Regime", "AIScore"]
    ].copy()
    styled = perf_df.style.format(
        {
            "Balance": "${:,.2f}",
            "Realized": "${:,.2f}",
            "Unrealized": "${:,.2f}",
            "WinRate": "{:.2%}",
            "AIScore": "{:.3f}",
        }
    ).map(
        lambda val: "color: #2ecc71;" if safe_float(val) > 0 else ("color: #e74c3c;" if safe_float(val) < 0 else ""),
        subset=["Realized", "Unrealized"],
    )
    st.markdown(
        styled.to_html(),
        unsafe_allow_html=True,
    )
    st.divider()


def api_get(endpoint, params=None, selected_symbol: str | None = None):
    base = get_api_base(selected_symbol or st.session_state.symbol)
    try:
        r = requests.get(f"{base}/{endpoint}", params=params, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.error(f"API Error: {exc}")
        return None


def api_post(endpoint, data=None, headers=None, selected_symbol: str | None = None):
    base = get_api_base(selected_symbol or st.session_state.symbol)
    try:
        r = requests.post(f"{base}/{endpoint}", json=data, headers=headers, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.error(f"API Error: {exc}")
        return None


status = api_get("status", selected_symbol=symbol)
hard_reset_feedback = st.session_state.pop("hard_reset_feedback", None)
if hard_reset_feedback:
    st.success(hard_reset_feedback)

# -- Sidebar Controls --
with st.sidebar:
    st.title("Engine Controls")

    if status:
        mode = status.get("mode", "UNKNOWN")
        mode_badge = "[PAPER]" if mode == "PAPER" else "[LIVE]"
        st.metric("Mode", f"{mode} {mode_badge}")
        st.metric("Engine", "Running" if status.get("engine_running") else "Stopped")
        st.metric("Daily PnL", f"${safe_float(status.get('daily_pnl')):.2f}")

        if mode == "PAPER":
            st.metric("Virtual Balance", f"${safe_float(status.get('virtual_balance')):.2f}")
            st.metric("Paper Realized", f"${safe_float(status.get('paper_realized_pnl')):.2f}")
            st.metric("Paper Unrealized", f"${safe_float(status.get('paper_unrealized_pnl')):.2f}")
        elif status.get("balance") is not None:
            st.metric("Live Balance", f"${safe_float(status.get('balance')):.2f}")

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
    st.caption(
        f"Aggression: {safe_float(thresholds.get('aggression_level', 1.0), 1.0):.2f} | "
        f"Shorts: {'ON' if bool(thresholds.get('allow_shorts', False)) else 'OFF'}"
    )
    new_conf = st.slider("Confidence", 0.40, 0.95, safe_float(thresholds.get("confidence_threshold", 0.70), 0.70), 0.05)
    new_risk = st.slider("Risk %", 0.2, 5.0, safe_float(thresholds.get("risk_per_trade", 0.01), 0.01) * 100, 0.1)
    if st.button("Update Thresholds"):
        api_post("thresholds", {"confidence_threshold": new_conf, "risk_per_trade": new_risk / 100})
        st.success("Updated")

    st.divider()
    auto_refresh_enabled = st.toggle("Auto-refresh (10s)", value=True)

    st.divider()
    st.subheader("Reset Paper Account")
    default_balance = settings.trading.paper_initial_balance
    try:
        env_default = safe_float(os.getenv("PAPER_INITIAL_BALANCE", default_balance), safe_float(default_balance))
    except ValueError:
        env_default = default_balance
    reset_balance = st.number_input(
        "Initial balance",
        min_value=0.0,
        value=env_default,
        step=100.0,
        format="%.2f",
        help="This value resets the paper wallet balance for the next cycle",
    )
    confirm_reset = st.checkbox(
        "I understand this will delete all paper history", key="reset_paper_ack"
    )
    if st.button("Reset Paper Account", type="primary", use_container_width=True):
        if not confirm_reset:
            st.error("Please acknowledge that all paper history will be deleted.")
        else:
            admin_token = os.getenv("ADMIN_TOKEN", "")
            if not admin_token:
                st.error("ADMIN_TOKEN is not configured in the environment.")
            else:
                response = api_post(
                    "admin/reset-paper",
                    {"initial_balance": safe_float(reset_balance)},
                    headers={"ADMIN_TOKEN": admin_token},
                )
                if response and response.get("ok"):
                    toast = getattr(st, "toast", None)
                    message = "Paper account reset triggered"
                    if callable(toast):
                        toast(message, icon="OK")
                    else:
                        st.success(message)
                    st.rerun()
                elif response:
                    st.error(response.get("detail", "Reset failed"))
                else:
                    st.error("Reset failed: no response from API.")

    st.divider()
    st.subheader("Danger Zone")
    st.markdown(
        """
        <style>
        [data-testid="stButton"][data-key="hard_reset_full_button"] > button {
            background-color: #d63031 !important;
            color: white !important;
            width: 100% !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    hard_reset_ack = st.checkbox(
        "I understand this will erase ALL trades and history.",
        key="hard_reset_ack",
    )
    if st.button("FULL HARD RESET", key="hard_reset_full_button", use_container_width=True):
        if not hard_reset_ack:
            st.error("Please confirm that you understand the full reset will erase everything.")
        else:
            admin_token = os.getenv("ADMIN_TOKEN", "")
            if not admin_token:
                st.error("ADMIN_TOKEN is not configured in the environment.")
            else:
                with st.spinner("Requesting full engine hard reset..."):
                    response = api_post(
                        "admin/hard-reset",
                        headers={"ADMIN_TOKEN": admin_token},
                    )
                if response and response.get("status") == "success":
                    success_msg = "Engine hard reset completed"
                    st.session_state.clear()
                    st.session_state["hard_reset_feedback"] = success_msg
                    st.experimental_rerun()
                elif response:
                    st.error(response.get("detail", "Hard reset failed"))
                else:
                    st.error("Hard reset failed: no response from API.")
    st.caption("This action cannot be undone.")


# -- Main --
def render_asset_dashboard(symbol: str, status: dict | None, auto_refresh_enabled: bool) -> None:
    st.title(f"TFT AI Trading Engine - {symbol}")
    
    stats = api_get("stats") or {}
    threshold_info = api_get("thresholds") or {}
    ai_score = api_get("ai/current-score") or {}
    df_ai_history = load_ai_metrics(symbol, limit=180)
    ai_history = df_ai_history.to_dict("records") if not df_ai_history.empty else (api_get("ai/history", {"limit": 180}) or [])
    agent_attribution = api_get("agent-attribution") or []
    df_risk_metrics = load_metrics(symbol, limit=300)
    risk_metrics = df_risk_metrics.to_dict("records") if not df_risk_metrics.empty else (api_get("risk-metrics", {"limit": 300}) or [])
    strategy_evolution = api_get("strategy-evolution", {"limit": 300}) or []
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Trades", stats.get("total_trades", 0))
    c2.metric("Win Rate", f"{safe_float(stats.get('win_rate')):.1%}")
    c3.metric("Total PnL", f"${safe_float(stats.get('total_pnl')):.2f}")
    c4.metric("Avg R", f"{safe_float(stats.get('avg_r_multiple')):.2f}R")
    c5.metric("W/L", f"{stats.get('winning_trades', 0)}/{stats.get('losing_trades', 0)}")
    c6.metric("Long/Short", f"{stats.get('long_trades', 0)}/{stats.get('short_trades', 0)}")

    a1, a2, a3 = st.columns(3)
    a1.metric("AI Score", f"{safe_float(ai_score.get('ai_score')):.2%}")
    a2.metric("AI Confidence", f"{safe_float(ai_score.get('confidence')):.2%}")
    a3.metric("AI Model", ai_score.get("model_version", "unknown"))
    
    r1, r2, r3, r4 = st.columns(4)
    status_payload = status or {}
    r1.metric("Aggression", f"{safe_float(status_payload.get('aggression_level', threshold_info.get('aggression_level', 1.0)), 1.0):.2f}")
    r2.metric("Shorts", "ON" if bool(status_payload.get("allow_shorts", threshold_info.get("allow_shorts", False))) else "OFF")
    r3.metric("Scaled Threshold", f"{safe_float(status_payload.get('threshold_after_scaling')):.3f}")
    latest_regime = status_payload.get("latest_regime") or {}
    r4.metric(
        "Regime",
        f"{latest_regime.get('trend', 'n/a')} / {latest_regime.get('volatility', 'n/a')}"
        if latest_regime
        else "n/a",
    )
    
    gauge_fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=safe_float(ai_score.get("confidence")) * 100.0,
            title={"text": "AI Confidence Gauge"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#3FA7D6"},
                "steps": [
                    {"range": [0, 50], "color": "#5A1A1A"},
                    {"range": [50, 70], "color": "#665500"},
                    {"range": [70, 100], "color": "#144B2B"},
                ],
            },
        )
    )
    gauge_fig.update_layout(height=260, template="plotly_dark")
    st.plotly_chart(gauge_fig, use_container_width=True)
    
    if ai_history:
        df_ai = pd.DataFrame(ai_history)
        if not df_ai.empty:
            fig_ai = go.Figure()
            fig_ai.add_trace(
                go.Scatter(
                    x=df_ai["timestamp"],
                    y=df_ai["ai_score"],
                    mode="lines",
                    name="AI Score",
                    line=dict(color="gold", width=2),
                )
            )
            fig_ai.add_trace(
                go.Scatter(
                    x=df_ai["timestamp"],
                    y=df_ai["confidence"],
                    mode="lines",
                    name="Confidence",
                    line=dict(color="deepskyblue", width=2),
                )
            )
            fig_ai.update_layout(
                title="AI Score History",
                yaxis_title="Score",
                template="plotly_dark",
                height=260,
            )
            st.plotly_chart(fig_ai, use_container_width=True)
    
    if status and status.get("mode") == "PAPER":
        realized = safe_float(status.get("paper_realized_pnl"))
        unrealized = safe_float(status.get("paper_unrealized_pnl"))
        virtual_balance = safe_float(status.get("virtual_balance"))
        p1, p2, p3 = st.columns(3)
        p1.metric("Paper Balance", f"${virtual_balance:.2f}")
        p2.metric("Realized PnL", f"${realized:.2f}")
        p3.metric("Unrealized PnL", f"${unrealized:.2f}")
    
    
    TAB_NAMES = [
        "?? PnL Curve",
        "?? Trades",
        "?? Predictions",
        "?? Learning",
        "?? Open Trade",
        "?? AI Analytics",
        "?? Agent Attribution",
        "??? Risk Metrics",
        "?? Strategy Evolution",
        "?? Decision Log",
    ]
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs(TAB_NAMES)
    
    with tab1:
        df = load_pnl(symbol)
        if df.empty:
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
            st.warning("Engine not producing trades — check monitor")
    
    with tab2:
        df = load_trades(symbol, limit=50)
        if df.empty:
            trades = api_get("trades", {"limit": 50}) or []
            df = pd.DataFrame(trades)
        if not df.empty:
            display_cols = [
                c for c in [
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
                ] if c in df.columns
            ]
            st.dataframe(
                df[display_cols] if display_cols else df,
                use_container_width=True,
                height=500,
            )
    
            for _, row in df.iterrows():
                if row.get("ai_reasoning"):
                    trade_id = row.get("trade_id", "unknown")
                    with st.expander(f"AI Reasoning: {trade_id}"):
                        st.text(row["ai_reasoning"])
        else:
            st.warning("Engine not producing trades — check monitor")
    
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
        open_positions_df = load_open_positions(symbol)
        if not open_positions_df.empty:
            st.subheader("Open Paper Positions")
            st.dataframe(open_positions_df, use_container_width=True)
        elif status and status.get("mode") == "PAPER":
            paper_positions = status.get("paper_positions") or []
            if paper_positions:
                st.subheader("Open Paper Positions")
                st.dataframe(pd.DataFrame(paper_positions), use_container_width=True)
            else:
                st.info("No open paper positions")
    
        if status and status.get("open_trade"):
            trade = status["open_trade"]
            st.subheader(f"Open: {trade['pair']}")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Side", str(trade.get("side", "BUY")))
            c2.metric("Entry", f"${safe_float(trade.get('entry_price')):.6f}")
            c3.metric("Stop", f"${safe_float(trade.get('stop_price')):.6f}")
            c4.metric("Target", f"${safe_float(trade.get('target_price')):.6f}")
            c5.metric("Confidence", f"{safe_float(trade.get('confidence')):.3f}")
            c6, c7 = st.columns(2)
            c6.metric("AI Score", f"{safe_float(trade.get('ai_score')):.3f}")
            c7.metric("AI Confidence", f"{safe_float(trade.get('ai_confidence')):.3f}")
    
            if st.button("Force Close Position", type="primary"):
                trade_ref = trade.get("id") or trade.get("trade_id")
                if trade_ref is not None:
                    api_post(f"trades/{trade_ref}/force_close")
                else:
                    api_post("control", {"action": "force_close"})
                st.rerun()
        else:
            st.info("No open position")
    
    with tab6:
        c1, c2 = st.columns([1, 2])
        with c1:
            period_days = st.selectbox("Period", options=[7, 30], index=0, key="ai_analytics_period")
        with c2:
            score_mode = st.radio(
                "Score View",
                options=["Show governance influence", "Show base AI score only"],
                horizontal=True,
                key="ai_analytics_mode",
            )
    
        analytics = api_get("ai/analytics", {"days": int(period_days)}) or {}
        timeline = analytics.get("timeline") or []
        if not timeline:
            st.info("No closed trades with AI score data in selected period.")
        else:
            df = pd.DataFrame(timeline)
            show_influence = score_mode == "Show governance influence"
    
            ai_fig = go.Figure()
            ai_fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["final_score"] if show_influence else df["before_score"],
                    mode="lines",
                    name="Final AI Score" if show_influence else "Base AI Score",
                    line=dict(color="#1f77b4", width=2),
                )
            )
            if show_influence:
                ai_fig.add_trace(
                    go.Scatter(
                        x=df["timestamp"],
                        y=df["before_score"],
                        mode="lines",
                        name="Before Governance",
                        line=dict(color="#ff7f0e", width=2, dash="dot"),
                    )
                )
            ai_fig.update_layout(
                title=f"AI Score Over Time ({period_days}d)",
                yaxis_title="AI Score",
                template="plotly_dark",
                height=320,
            )
            st.plotly_chart(ai_fig, use_container_width=True)
    
            impact = analytics.get("impact") or {}
            impact_fig = go.Figure(
                data=[
                    go.Bar(
                        x=["Before Governance", "Final Score"],
                        y=[
                            safe_float(impact.get("avg_before", 0.0)),
                            safe_float(impact.get("avg_final", 0.0)),
                        ],
                        marker_color=["#ff7f0e", "#1f77b4"],
                    )
                ]
            )
            impact_fig.update_layout(
                title="Governance Impact (Average Score)",
                yaxis_title="Score",
                template="plotly_dark",
                height=280,
            )
            st.plotly_chart(impact_fig, use_container_width=True)
    
            hist_fig = go.Figure(
                data=[
                    go.Histogram(
                        x=df["confidence"],
                        nbinsx=20,
                        marker_color="#00cc96",
                        opacity=0.85,
                        name="Confidence",
                    )
                ]
            )
            hist_fig.update_layout(
                title="Confidence Distribution",
                xaxis_title="Confidence",
                yaxis_title="Count",
                template="plotly_dark",
                height=280,
            )
            st.plotly_chart(hist_fig, use_container_width=True)
    
            overlay = make_subplots(specs=[[{"secondary_y": True}]])
            overlay.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["final_score"] if show_influence else df["before_score"],
                    mode="lines",
                    name="AI Score",
                    line=dict(color="#ffd166", width=2),
                ),
                secondary_y=False,
            )
            overlay.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["drawdown"],
                    mode="lines",
                    name="Drawdown",
                    line=dict(color="#ef476f", width=2),
                ),
                secondary_y=True,
            )
            overlay.update_layout(
                title="Drawdown Overlay vs AI Score",
                template="plotly_dark",
                height=320,
            )
            overlay.update_yaxes(title_text="AI Score", secondary_y=False)
            overlay.update_yaxes(title_text="Drawdown ($)", secondary_y=True)
            st.plotly_chart(overlay, use_container_width=True)
    
    with tab7:
        if not agent_attribution:
            st.info("No agent attribution data yet.")
        else:
            df = pd.DataFrame(agent_attribution)
            st.dataframe(df, use_container_width=True)
    
            pnl_fig = go.Figure(
                data=[
                    go.Bar(
                        x=df["agent"],
                        y=df["total_pnl"],
                        marker_color=["#06d6a0", "#118ab2", "#ffd166", "#ef476f"][: len(df)],
                        name="Cumulative PnL",
                    )
                ]
            )
            pnl_fig.update_layout(
                title="Cumulative PnL by Agent",
                xaxis_title="Agent",
                yaxis_title="PnL ($)",
                template="plotly_dark",
                height=320,
            )
            st.plotly_chart(pnl_fig, use_container_width=True)
    
            win_fig = go.Figure(
                data=[
                    go.Bar(
                        x=df["agent"],
                        y=df["win_rate"],
                        marker_color="#90e0ef",
                        name="Win Rate",
                    )
                ]
            )
            win_fig.update_layout(
                title="Win Rate by Agent",
                xaxis_title="Agent",
                yaxis_title="Win Rate",
                template="plotly_dark",
                height=320,
            )
            win_fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(win_fig, use_container_width=True)
    
    with tab8:
        if not risk_metrics:
            st.info("No risk metrics snapshots yet.")
        else:
            df = pd.DataFrame(risk_metrics)
            if not df.empty:
                sharpe_fig = go.Figure()
                sharpe_fig.add_trace(
                    go.Scatter(
                        x=df["timestamp"],
                        y=df["sharpe"],
                        mode="lines",
                        line=dict(color="#00b4d8", width=2),
                        name="Sharpe",
                    )
                )
                sharpe_fig.update_layout(
                    title="Sharpe Ratio",
                    template="plotly_dark",
                    height=280,
                )
                st.plotly_chart(sharpe_fig, use_container_width=True)
    
                sortino_fig = go.Figure()
                sortino_fig.add_trace(
                    go.Scatter(
                        x=df["timestamp"],
                        y=df["sortino"],
                        mode="lines",
                        line=dict(color="#ffd166", width=2),
                        name="Sortino",
                    )
                )
                sortino_fig.update_layout(
                    title="Sortino Ratio",
                    template="plotly_dark",
                    height=280,
                )
                st.plotly_chart(sortino_fig, use_container_width=True)
    
                dd_fig = go.Figure()
                dd_fig.add_trace(
                    go.Scatter(
                        x=df["timestamp"],
                        y=df["max_drawdown"],
                        mode="lines",
                        line=dict(color="#ef476f", width=2),
                        name="Max Drawdown",
                    )
                )
                dd_fig.update_layout(
                    title="Drawdown Curve",
                    yaxis_title="Drawdown",
                    template="plotly_dark",
                    height=300,
                )
                st.plotly_chart(dd_fig, use_container_width=True)
    
    with tab9:
        if not strategy_evolution:
            st.info("No strategy evolution snapshots yet.")
        else:
            df = pd.DataFrame(strategy_evolution)
            st.dataframe(df.tail(50), use_container_width=True)
    
            weights_fig = go.Figure()
            weights_fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["tft_weight"],
                    mode="lines",
                    name="TFT Weight",
                    line=dict(color="#00b4d8", width=2),
                )
            )
            weights_fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["xgb_weight"],
                    mode="lines",
                    name="XGB Weight",
                    line=dict(color="#ffd166", width=2),
                )
            )
            weights_fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["ppo_weight"],
                    mode="lines",
                    name="PPO Weight",
                    line=dict(color="#ef476f", width=2),
                )
            )
            weights_fig.update_layout(
                title="Strategy Weights Over Time",
                yaxis_title="Weight",
                template="plotly_dark",
                height=320,
            )
            st.plotly_chart(weights_fig, use_container_width=True)
    
            conf_fig = go.Figure()
            conf_fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["confidence_threshold"],
                    mode="lines",
                    name="Confidence Threshold",
                    line=dict(color="#90e0ef", width=2),
                )
            )
            conf_fig.update_layout(
                title="Confidence Threshold Evolution",
                yaxis_title="Threshold",
                template="plotly_dark",
                height=280,
            )
            st.plotly_chart(conf_fig, use_container_width=True)
    
    with tab10:
        decision_events = api_get("decision-events", {"limit": 50}) or []
        if not decision_events:
            st.info("No decision events recorded yet. Events appear after the engine runs at least one cycle.")
        else:
            df_de = pd.DataFrame(decision_events)
            # Summary metrics
            total_events = len(df_de)
            no_trade_count = len(df_de[df_de["status"] == "no_trade"])
            trade_count = len(df_de[df_de["status"] == "trade_opened"])
            de1, de2, de3 = st.columns(3)
            de1.metric("Total Cycles", total_events)
            de2.metric("Trades Opened", trade_count)
            de3.metric("No-Trade Cycles", no_trade_count)
    
            # Reason breakdown for no-trade events
            no_trade_df = df_de[df_de["status"] == "no_trade"]
            if not no_trade_df.empty and "reason" in no_trade_df.columns:
                reason_counts = no_trade_df["reason"].value_counts()
                reason_fig = go.Figure(
                    data=[go.Bar(x=reason_counts.index.tolist(), y=reason_counts.values.tolist(), marker_color="#ffd166")]
                )
                reason_fig.update_layout(
                    title="No-Trade Reasons",
                    xaxis_title="Reason",
                    yaxis_title="Count",
                    template="plotly_dark",
                    height=280,
                )
                st.plotly_chart(reason_fig, use_container_width=True)
    
            display_cols = [
                c for c in [
                    "timestamp", "status", "reason", "best_pair", "best_score",
                    "best_ai_score", "best_confidence", "best_prob_up", "best_prob_down",
                    "regime", "candidates_evaluated", "candidates_valid",
                ] if c in df_de.columns
            ]
            st.dataframe(df_de[display_cols], use_container_width=True, height=400)
    
    model = api_get("model-info")
    if model and model.get("active_model"):
        with st.expander("Active Model"):
            st.json(model["active_model"])
    
    st.markdown("---")
    st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    if auto_refresh_enabled:
        import time
    
        time.sleep(10)
        st.rerun()
    


render_portfolio_overview(PAIRS)
render_asset_dashboard(symbol, status, auto_refresh_enabled)
