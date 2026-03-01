# TFT AI Trading Engine

Production-grade AI trading engine using Temporal Fusion Transformer (TFT) for multi-horizon price forecasting with KuCoin Spot execution.

## Architecture

```
tft-trading-engine/
├── config/          # Configuration & environment
├── data/            # Data fetching, processing, feature engineering
├── models/          # TFT model definition, training, inference
├── engine/          # AI decision engine, signal generation
├── execution/       # KuCoin API integration, order management
├── dashboard/       # FastAPI + Streamlit admin UI
├── backtesting/     # Walk-forward backtester, Monte Carlo
├── logs/            # Structured JSON log storage
├── utils/           # Shared utilities
├── scripts/         # Training, backtesting, deployment scripts
├── tests/           # Unit and integration tests
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

## Features

- **TFT Forecasting**: Multi-horizon price prediction with attention-based learning
- **Smart Pair Selection**: AI selects optimal pair from top 30 USDT pairs
- **Dynamic Risk Management**: ATR-based stops, trailing stops, confidence-based exits
- **Self-Improving Loop**: Post-trade analysis, threshold adjustment, weekly retraining
- **Admin Dashboard**: Real-time monitoring, PnL curves, kill switches
- **Safety Layer**: Circuit breakers, max loss limits, API key encryption
- **Backtesting**: Walk-forward validation, Monte Carlo simulation, slippage modeling

## Runtime Requirements

- Python 3.11.x (recommended)
- CUDA-compatible PyTorch

## Quick Start

### 1. Environment Setup

```bash
cp .env.example .env
# Edit .env with your KuCoin API credentials
```

### 2. Docker Deployment

```bash
docker-compose up -d
```

### Docker Runtime Notes

- `ENGINE_MODE` controls the entrypoint reset behavior (PAPER only) and `TRADING_MODE` controls live vs paper execution.
- The engine exposes a lightweight `/health` endpoint on port 8000 for Docker healthchecks.
- State is persisted to the host in `./state` (engine DB + markers) and logs in `./logs`.
- In PAPER mode, the entrypoint wipes `/app/state/tft_engine.db` only on first boot, then records `.paper_init_done`.
- Use `docker-compose.override.yml` for development defaults (PAPER mode + healthcheck).

### 3. Manual Setup (Windows)

1. Install Python 3.11 from python.org.
2. Create a virtual environment:

```powershell
py -3.11 -m venv .venv
```

3. Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

4. Install requirements:

```powershell
pip install -r requirements.txt
```

Then run the engine workflow:

```bash
# Initialize database
python scripts/init_db.py

# Fetch historical data
python scripts/fetch_history.py

# Train model
python scripts/train_model.py

# Run backtester
python scripts/run_backtest.py

# Start engine (only if backtest passes)
python scripts/run_engine.py

# Start dashboard
python scripts/run_dashboard.py
```

## Configuration

All configuration is in `config/settings.py` and can be overridden via environment variables.

Key settings:
- `RISK_PER_TRADE`: Default 1% of balance
- `CONFIDENCE_THRESHOLD`: Default 0.70
- `MAX_DAILY_LOSS_PCT`: Default 3%
- `MAX_CONSECUTIVE_LOSSES`: Default 5
- `MIN_SHARPE_RATIO`: Default 1.5
- `MAX_DRAWDOWN_PCT`: Default 15%

## Safety

- API keys are encrypted at rest using Fernet
- All secrets via environment variables
- Circuit breaker on abnormal volatility
- Emergency kill switch in dashboard
- Max daily loss protection
- Position reconciliation on restart

## Reset Paper Account

- **CLI:** `python scripts/reset_paper_account.py --initial-balance 10000 --confirm`  
  Performs a full paper reset (trades, predictions, metrics, sqlite snapshot, unrealized PnL) and signals the engine to reload. Omitting `--confirm` runs a dry-run that prints what would be cleared.
- **API:** `POST /admin/reset-paper` with JSON `{"initial_balance": 10000}` and header `ADMIN_TOKEN`. The FastAPI endpoint returns confirms, deleted row counts, and sqlite cleanup details so the dashboard and automation can react.
- **Dashboard:** The Engine Controls sidebar now exposes a `Reset Paper Account` button, acknowledgement checkbox, and balance input to trigger the API via JavaScript. On success it shows a toast and refreshes the widgets.

Set `PAPER_INITIAL_BALANCE` (defaults to 10 000) to change the reset target across the CLI, engine, and dashboard input. Keep `ADMIN_TOKEN` synchronized between `.env`, Docker Compose, and any automation to call the admin endpoint safely.
