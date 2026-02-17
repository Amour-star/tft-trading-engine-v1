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

### 3. Manual Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

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
