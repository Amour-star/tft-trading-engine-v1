"""
FastAPI application for TFT Trading Engine v2.
"""
from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from tft_engine.api.routes.ai import router as ai_router
from tft_engine.api.routes.metrics import router as metrics_router
from tft_engine.api.routes.opus import router as opus_router
from tft_engine.api.routes.trades import router as trades_router
from tft_engine.config import config

logging.basicConfig(
    level=getattr(logging, config.log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(title="TFT Trading Engine v2", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(ai_router)
app.include_router(opus_router)
app.include_router(trades_router)
app.include_router(metrics_router)


@app.get("/health")
def health():
    gpu_available = False
    try:
        import torch

        gpu_available = bool(torch.cuda.is_available())
    except Exception:
        gpu_available = False

    return {
        "status": "ok",
        "ai_enabled": bool(config.ai_enabled),
        "governance_enabled": bool(config.opus_enabled),
        "db": "sqlite",
        "gpu": gpu_available,
    }
