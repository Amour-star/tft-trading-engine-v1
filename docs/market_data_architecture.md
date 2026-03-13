# Institutional Market Data Architecture

```text
KuCoin REST (/api/v1/timestamp, /api/v1/bullet-public)
        |
        v
services/market_data/kucoin_rest.py
        |
        +----> services/market_data/heartbeat_monitor.py (2s checks)
        |           - API reachable
        |           - WS active
        |           - price freshness per symbol <= 5s
        |           - on failure => TRADING_HALTED signal
        |
        v
services/market_data/kucoin_ws.py (ticker stream)
        |
        v
services/market_data/data_validator.py
        - reject if drift > 3%
        - reject if timestamp older than 5s
        - reject if volume == 0
        |
        v
Redis
  - channel: market:ticker
  - key: market:ticker:{SYMBOL}
  - halt key: market:halt:{SYMBOL}
  - control channel: market:control
        |
        v
data/fetcher.py (engine consumer)
  - reads/subscribes market:ticker only
  - no synthetic / last-known fallback
  - on unavailable: RuntimeError("Live market data unavailable")
        |
        v
Engine + Executors + Monitor
  - pause trading
  - force close risky/open positions
  - critical log: TRADING_HALTED

Observability endpoints
  - /market/status
  - /market/source
  - /market/latency
```
