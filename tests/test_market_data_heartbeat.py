from __future__ import annotations

import time

from services.market_data.heartbeat_monitor import HeartbeatMonitor


class _RestStub:
    def is_reachable(self) -> bool:
        return True


def test_heartbeat_treats_rest_fallback_as_healthy() -> None:
    halted: list[tuple[str, dict]] = []
    recovered: list[dict] = []

    now = time.time()
    monitor = HeartbeatMonitor(
        symbols=["BTC-USDT"],
        rest_client=_RestStub(),
        ws_active_provider=lambda: False,
        latest_ticker_provider=lambda: {
            "BTC-USDT": {
                "timestamp": now,
            }
        },
        on_halt=lambda reason, heartbeat: halted.append((reason, heartbeat)),
        on_recover=lambda heartbeat: recovered.append(heartbeat),
        interval_seconds=0.05,
        max_age_seconds=5.0,
    )

    monitor.start()
    time.sleep(0.12)
    monitor.stop()

    assert monitor.is_healthy is True
    assert halted == []
    assert recovered == []
    assert monitor.latest_heartbeat["price_recent"] is True
    assert monitor.latest_heartbeat["ws_active"] is False
    assert monitor.latest_heartbeat["transport_degraded"] is True


def test_heartbeat_halts_when_prices_are_stale() -> None:
    halted: list[tuple[str, dict]] = []

    now = time.time()
    monitor = HeartbeatMonitor(
        symbols=["BTC-USDT"],
        rest_client=_RestStub(),
        ws_active_provider=lambda: False,
        latest_ticker_provider=lambda: {
            "BTC-USDT": {
                "timestamp": now - 10.0,
            }
        },
        on_halt=lambda reason, heartbeat: halted.append((reason, heartbeat)),
        on_recover=None,
        interval_seconds=0.05,
        max_age_seconds=5.0,
    )

    monitor.start()
    time.sleep(0.12)
    monitor.stop()

    assert monitor.is_healthy is False
    assert len(halted) >= 1
    assert halted[0][0] == "market_data_unhealthy"
    assert halted[0][1]["price_recent"] is False
