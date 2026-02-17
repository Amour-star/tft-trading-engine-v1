"""
Temporal Fusion Transformer (TFT) model for multi-horizon price forecasting.
Uses PyTorch Forecasting for model construction and training.
"""
from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger

try:
    import lightning.pytorch as pl
    from pytorch_forecasting import (
        TemporalFusionTransformer,
        TimeSeriesDataSet,
        GroupNormalizer,
    )
    from pytorch_forecasting.metrics import QuantileLoss
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    HAS_TFT = True
except ImportError:
    HAS_TFT = False
    logger.warning("pytorch-forecasting not installed. Model training/inference disabled.")

from config.settings import settings
from data.features import get_feature_columns, get_categorical_columns


MODEL_DIR = Path(settings.log_dir).parent / "saved_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def _prepare_dataframe(
    df: pd.DataFrame,
    prediction_length: Optional[int] = None,
) -> pd.DataFrame:
    """Prepare raw DataFrame by adding time_idx, target, and fixing dtypes."""
    cfg = settings.model
    pred_len = prediction_length or cfg.prediction_length

    df = df.copy()
    df.sort_values(["pair", "timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Create time index per group (sequential integer)
    df["time_idx"] = df.groupby("pair").cumcount()

    # Target: future return (log return over prediction horizon)
    df["target"] = df.groupby("pair")["close"].transform(
        lambda x: np.log(x.shift(-pred_len) / x)
    )
    df.dropna(subset=["target"], inplace=True)

    # Ensure categorical columns are string type
    cat_cols = [c for c in get_categorical_columns() if c in df.columns]
    for col in cat_cols:
        df[col] = df[col].astype(str)

    return df


def prepare_tft_dataset(
    df: pd.DataFrame,
    encoder_length: Optional[int] = None,
    prediction_length: Optional[int] = None,
    training: bool = True,
    processed_df: Optional[pd.DataFrame] = None,
) -> "TimeSeriesDataSet":
    """
    Convert feature DataFrame into a TimeSeriesDataSet for TFT.

    The DataFrame must have columns:
      - timestamp, close, and all feature columns
      - A 'pair' column for group identification

    If processed_df is provided, it will be used directly (must already
    contain time_idx and target columns).
    """
    if not HAS_TFT:
        raise RuntimeError("pytorch-forecasting is required for TFT dataset creation")

    cfg = settings.model
    enc_len = encoder_length or cfg.encoder_length
    pred_len = prediction_length or cfg.prediction_length

    if processed_df is not None:
        df = processed_df
    else:
        df = _prepare_dataframe(df, pred_len)

    cat_cols = [c for c in get_categorical_columns() if c in df.columns]

    # Continuous features
    cont_features = [c for c in get_feature_columns() if c in df.columns]

    # Time-varying known (temporal encodings)
    known_reals = [c for c in ["hour_sin", "hour_cos", "dow_sin", "dow_cos"] if c in df.columns]

    # Time-varying unknown (most features - not known in advance)
    unknown_reals = [c for c in cont_features if c not in known_reals]

    if training:
        # Use 80% of each group's time range for training
        max_time_idx = df.groupby("pair")["time_idx"].transform("max")
        training_cutoff = int(max_time_idx.iloc[0] * 0.8)

        dataset = TimeSeriesDataSet(
            df[lambda x: x.time_idx <= training_cutoff],
            time_idx="time_idx",
            target="target",
            group_ids=["pair"],
            min_encoder_length=enc_len // 2,
            max_encoder_length=enc_len,
            min_prediction_length=1,
            max_prediction_length=pred_len,
            static_categoricals=[],
            static_reals=[],
            time_varying_known_categoricals=[c for c in cat_cols if c in ["session"]],
            time_varying_known_reals=known_reals,
            time_varying_unknown_categoricals=[c for c in cat_cols if c not in ["session"]],
            time_varying_unknown_reals=unknown_reals + ["target"],
            target_normalizer=GroupNormalizer(groups=["pair"]),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )
    else:
        # For inference, use entire dataset
        dataset = TimeSeriesDataSet(
            df,
            time_idx="time_idx",
            target="target",
            group_ids=["pair"],
            min_encoder_length=enc_len // 2,
            max_encoder_length=enc_len,
            min_prediction_length=1,
            max_prediction_length=pred_len,
            static_categoricals=[],
            static_reals=[],
            time_varying_known_categoricals=[c for c in cat_cols if c in ["session"]],
            time_varying_known_reals=known_reals,
            time_varying_unknown_categoricals=[c for c in cat_cols if c not in ["session"]],
            time_varying_unknown_reals=unknown_reals + ["target"],
            target_normalizer=GroupNormalizer(groups=["pair"]),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

    return dataset


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_tft_model(
    dataset: "TimeSeriesDataSet",
) -> "TemporalFusionTransformer":
    """Build a TFT model from a dataset."""
    if not HAS_TFT:
        raise RuntimeError("pytorch-forecasting required")

    cfg = settings.model
    model = TemporalFusionTransformer.from_dataset(
        dataset,
        learning_rate=cfg.learning_rate,
        hidden_size=cfg.hidden_size,
        attention_head_size=cfg.attention_head_size,
        dropout=cfg.dropout,
        hidden_continuous_size=cfg.hidden_size // 2,
        output_size=7,  # 7 quantiles for uncertainty estimation
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=5,
    )
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_tft(
    df: pd.DataFrame,
    model_name: Optional[str] = None,
) -> Tuple[str, Dict[str, float]]:
    """
    Train TFT model on prepared data.

    Returns
    -------
    (model_version, metrics_dict)
    """
    if not HAS_TFT:
        raise RuntimeError("pytorch-forecasting required")

    cfg = settings.model
    version = model_name or f"tft_v{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    save_path = MODEL_DIR / version

    logger.info(f"Training TFT model: {version}")

    # Preprocess data once (adds time_idx, target, fixes dtypes)
    processed_df = _prepare_dataframe(df)

    # Build dataset
    training_dataset = prepare_tft_dataset(df, training=True, processed_df=processed_df)

    # Validation dataset (uses full processed df with target column)
    val_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        processed_df,
        predict=True,
        stop_randomization=True,
    )

    train_loader = training_dataset.to_dataloader(
        train=True, batch_size=cfg.batch_size, num_workers=0
    )
    val_loader = val_dataset.to_dataloader(
        train=False, batch_size=cfg.batch_size * 2, num_workers=0
    )

    # Callbacks
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        min_delta=1e-5,
        mode="min",
    )
    checkpoint = ModelCheckpoint(
        dirpath=str(save_path),
        filename="best-{epoch}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator="auto",
        gradient_clip_val=0.1,
        callbacks=[early_stop, checkpoint],
        enable_progress_bar=True,
        log_every_n_steps=10,
    )

    # Build model
    model = build_tft_model(training_dataset)

    # Train
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Load best checkpoint
    best_model_path = checkpoint.best_model_path
    best_model = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    # Compute metrics
    val_metrics = trainer.validate(best_model, dataloaders=val_loader)
    metrics = {
        "training_loss": float(trainer.callback_metrics.get("train_loss", 0)),
        "validation_loss": float(val_metrics[0].get("val_loss", 0)) if val_metrics else 0,
        "best_epoch": int(early_stop.stopped_epoch or cfg.max_epochs),
    }

    # Save version info
    info_path = save_path / "info.txt"
    with open(info_path, "w") as f:
        f.write(f"version: {version}\n")
        f.write(f"trained_at: {datetime.utcnow().isoformat()}\n")
        f.write(f"metrics: {metrics}\n")

    logger.info(f"Model {version} trained. Val loss: {metrics['validation_loss']:.6f}")
    return version, metrics


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

class TFTPredictor:
    """Loads a trained TFT model and runs inference."""

    def __init__(self, model_version: Optional[str] = None) -> None:
        self.model: Optional[TemporalFusionTransformer] = None
        self.model_version: str = ""
        if model_version:
            self.load(model_version)

    def load(self, model_version: str) -> None:
        """Load a saved model."""
        if not HAS_TFT:
            raise RuntimeError("pytorch-forecasting required")

        model_path = MODEL_DIR / model_version
        ckpt_files = list(model_path.glob("*.ckpt"))
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoint found in {model_path}")

        best_ckpt = sorted(ckpt_files, key=lambda p: p.stat().st_mtime, reverse=True)[0]
        # map_location handles CUDA→CPU fallback when no GPU is available
        map_loc = None if torch.cuda.is_available() else torch.device("cpu")
        self.model = TemporalFusionTransformer.load_from_checkpoint(
            str(best_ckpt), map_location=map_loc,
        )
        self.model.eval()
        self.model_version = model_version
        logger.info(f"Loaded model: {model_version} from {best_ckpt}")

    def predict(
        self,
        df: pd.DataFrame,
        pair: str,
    ) -> Dict[str, Any]:
        """
        Run inference on latest data for a single pair.

        Returns
        -------
        dict with keys:
            prob_up, prob_down, expected_move, confidence,
            forecast_vector, attention_weights
        """
        if self.model is None:
            raise RuntimeError("No model loaded")

        df = df[df["pair"] == pair].copy()
        if len(df) < settings.model.encoder_length + 5:
            return self._empty_prediction()

        # Prepare dataset for prediction
        try:
            dataset = prepare_tft_dataset(df, training=False)
            dataloader = dataset.to_dataloader(
                train=False, batch_size=1, num_workers=0
            )

            # Get predictions — use mode="quantiles" for full uncertainty output
            # model.predict() internally creates a Trainer with no_grad
            raw_preds = self.model.predict(
                dataloader, mode="quantiles", return_x=False,
            )

            # raw_preds shape depends on mode:
            #   3D (samples, horizons, quantiles) – full quantile output
            #   2D (samples, horizons)            – point forecast (median)
            raw_np = raw_preds.cpu().numpy()
            last_pred = raw_np[-1]  # Most recent sample

            if raw_np.ndim == 3 and raw_np.shape[2] >= 7:
                # Full quantile output
                median_forecast = last_pred[:, 3]     # 0.5 quantile
                lower_bound = last_pred[:, 0]         # 0.02 quantile
                upper_bound = last_pred[:, 6]         # 0.98 quantile

                positive_quantiles = sum(
                    1 for i in range(7) if float(np.nanmean(last_pred[:, i])) > 0
                )
                prob_up = positive_quantiles / 7.0
            else:
                # Point forecast only (2D) — treat as median
                median_forecast = last_pred  # shape: (horizons,)

                # Use variance across the forecast horizon as uncertainty proxy
                horizon_std = float(np.std(median_forecast))
                lower_bound = median_forecast - 2 * horizon_std
                upper_bound = median_forecast + 2 * horizon_std

                # Direction probability: use valid (non-NaN) samples
                valid_mask = ~np.isnan(raw_np).any(axis=1)
                if valid_mask.sum() > 0:
                    valid_means = np.nanmean(raw_np[valid_mask], axis=1)
                    prob_up = float((valid_means > 0).mean())
                else:
                    prob_up = 0.5

            prob_down = 1.0 - prob_up
            expected_move = float(np.nanmean(median_forecast))
            forecast_std = float(np.std(median_forecast))

            # Confidence: combine multiple signals for a robust score
            spread = float(np.mean(upper_bound - lower_bound))
            mean_abs_forecast = float(np.mean(np.abs(median_forecast))) + 1e-8

            # Component 1: Quantile tightness (30% weight)
            # How tight is the spread relative to the forecast?  Use log scale
            # so extreme spreads don't dominate.
            raw_uncertainty = spread / mean_abs_forecast
            tightness = max(0.0, min(1.0, 1.0 / (1.0 + raw_uncertainty * 0.5)))

            # Component 2: Directional agreement (40% weight)
            # How many quantiles agree on the direction?
            direction_score = max(prob_up, prob_down)  # 0.5 = no agreement, 1.0 = full agreement

            # Component 3: Forecast consistency (30% weight)
            # Does the median forecast stay in the same direction across horizons?
            if len(median_forecast) > 1:
                sign_changes = np.sum(np.diff(np.sign(median_forecast)) != 0)
                consistency = 1.0 - sign_changes / (len(median_forecast) - 1)
            else:
                consistency = 0.5

            # Weighted combination
            confidence = (
                0.30 * tightness +
                0.40 * direction_score +
                0.30 * consistency
            )
            confidence = round(max(0.1, min(1.0, confidence)), 4)

            # Slight penalty when forecast magnitude is near zero
            if mean_abs_forecast < 1e-6:
                confidence = min(confidence, 0.35)

            return {
                "prob_up": round(prob_up, 4),
                "prob_down": round(prob_down, 4),
                "expected_move": round(expected_move, 6),
                "confidence": round(confidence, 4),
                "forecast_vector": median_forecast.tolist(),
                "lower_bound": lower_bound.tolist(),
                "upper_bound": upper_bound.tolist(),
                "model_version": self.model_version,
            }

        except Exception as e:
            logger.error(f"Prediction error for {pair}: {e}")
            return self._empty_prediction()

    def get_attention_weights(self) -> Optional[Dict[str, Any]]:
        """Extract attention weights from last prediction for explainability."""
        if self.model is None:
            return None
        try:
            interpretation = self.model.interpret_output(
                self.model.predict(None, return_x=True), reduction="sum"
            )
            return {
                "attention_weights": interpretation.get("attention", None),
                "variable_importance": interpretation.get("static_variables", None),
                "encoder_importance": interpretation.get("encoder_variables", None),
                "decoder_importance": interpretation.get("decoder_variables", None),
            }
        except Exception:
            return None

    @staticmethod
    def _empty_prediction() -> Dict[str, Any]:
        return {
            "prob_up": 0.5,
            "prob_down": 0.5,
            "expected_move": 0.0,
            "confidence": 0.0,
            "forecast_vector": [],
            "lower_bound": [],
            "upper_bound": [],
            "model_version": "",
        }

    @staticmethod
    def list_models() -> List[str]:
        """List all saved model versions."""
        if not MODEL_DIR.exists():
            return []
        return sorted([d.name for d in MODEL_DIR.iterdir() if d.is_dir()])
