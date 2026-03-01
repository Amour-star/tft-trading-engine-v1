# TFT Trading Engine v2

## Run API

```bash
uvicorn tft_engine.api.app:app --host 0.0.0.0 --port 8000
```

## Run Dashboard

```bash
streamlit run tft_engine/dashboard/ui.py
```

## Train Models

```bash
python -m tft_engine.scripts.train_supervised
python -m tft_engine.scripts.train_rl
```

## Daily Retraining

```bash
python -m tft_engine.scripts.retrain_daily
```

## Notes

- Database: `sqlite:///data/tft.db`
- Supervised model: `models/supervised/latest_xgb.pkl`
- RL model: `models/rl/latest_rl.zip`
- Model auto-reload: every 300 seconds via `model_registry` polling.

## Opus + Cost Control

New endpoints:

- `POST /api/ai/opus/analyze`
- `GET /api/ai/opus/usage`
- `POST /api/ai/score/combined`
- `POST /api/ai/score/live`

Key environment variables:

- `OPUS_ENABLED=true`
- `ANTHROPIC_API_KEY=...`
- `OPUS_MODEL=claude-opus-4-1`
- `LLM_DAILY_BUDGET_USD=10`
- `LLM_MONTHLY_BUDGET_USD=200`
- `LLM_MAX_REQUEST_COST_USD=1`
- `LLM_MAX_INPUT_TOKENS=4000`
- `LLM_MAX_OUTPUT_TOKENS=800`
