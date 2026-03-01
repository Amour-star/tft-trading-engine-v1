"""
Trade open/close endpoints.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from tft_engine.api.runtime import get_runtime

router = APIRouter(prefix="/api/trades", tags=["trades"])


class OpenTradeRequest(BaseModel):
    symbol: str
    side: Literal["BUY", "SELL"] = "BUY"
    entry_price: float = Field(gt=0)
    current_price: Optional[float] = None
    account_balance: float = Field(default=10_000.0, gt=0)
    fees: float = Field(default=0.0, ge=0)
    rsi: float = 50.0
    ema_20: float = 0.0
    ema_50: float = 0.0
    atr: float = 0.0
    volatility: float = 0.0
    volume: float = 0.0
    volume_change: float = 0.0
    macd: float = 0.0
    market_regime: str = "sideways"
    timestamp: Optional[datetime] = None


class CloseTradeRequest(BaseModel):
    exit_price: float = Field(gt=0)
    fees: float = Field(default=0.0, ge=0)


@router.get("/open")
def list_open_trades():
    runtime = get_runtime()
    executor = runtime["executor"]
    trades = executor.get_open_trades()
    return [
        {
            "id": t.id,
            "symbol": t.symbol,
            "side": t.side,
            "entry_price": t.entry_price,
            "quantity": t.quantity,
            "fees": t.fees,
            "cost_with_fees": (t.entry_price * t.quantity) + (t.fees or 0.0),
            "ai_confidence": t.ai_confidence,
            "ai_score": t.ai_score,
            "model_version": t.model_version,
            "open_timestamp": t.open_timestamp.isoformat(),
        }
        for t in trades
    ]


@router.post("/open")
def open_trade(payload: OpenTradeRequest):
    runtime = get_runtime()
    controller = runtime["ai_controller"]
    snapshot = payload.model_dump()
    if snapshot.get("current_price") is None:
        snapshot["current_price"] = snapshot["entry_price"]
    snapshot["hour"] = (payload.timestamp or datetime.utcnow()).hour
    snapshot["day_of_week"] = (payload.timestamp or datetime.utcnow()).weekday()

    try:
        result = controller.evaluate_and_open_trade(
            market_snapshot=snapshot,
            account_balance=payload.account_balance,
            side=payload.side,
            fees=payload.fees,
        )
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to open trade: {exc}") from exc


@router.post("/{trade_id}/close")
def close_trade(trade_id: int, payload: CloseTradeRequest):
    runtime = get_runtime()
    executor = runtime["executor"]
    try:
        trade = executor.close_trade(trade_id, exit_price=payload.exit_price, fees=payload.fees)
        return {
            "trade_id": trade.id,
            "symbol": trade.symbol,
            "pnl": trade.pnl,
            "win": trade.win,
            "close_timestamp": trade.close_timestamp.isoformat() if trade.close_timestamp else None,
        }
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to close trade: {exc}") from exc
