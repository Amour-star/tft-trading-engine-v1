"""
Temporal Fusion Transformer (TFT) model for multi-horizon price forecasting.
Uses PyTorch Forecasting for model construction and training.
"""
from __future__ import annotations

import logging
import os
import warnings
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

try:
    import torch

    HAS_TORCH = True
except Exception:
    torch = None  # type: ignore[assignment]
    HAS_TORCH = False

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_forecasting import GroupNormalizer, TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.metrics import QuantileLoss

    HAS_TFT = True
except Exception:
    HAS_TFT = False
    if not HAS_TORCH:
        logger.warning("torch not installed. Model training/inference disabled.")
    else:
        logger.warning("pytorch-forecasting not installed. Model training/inference disabled.")

from config.settings import XRP_ONLY_SYMBOL, settings
from data.features import get_categorical_columns, get_feature_columns

if HAS_TORCH and hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("medium")


MODEL_DIR = Path(settings.log_dir).parent / "saved_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def _configure_lightning_warning_filters() -> None:
    """
    Suppress known non-actionable Lightning warnings in this service runtime.
    """
    warnings.filterwarnings(
        "ignore",
        message=r"Attribute 'loss' is an instance of `nn\.Module` and is already saved during checkpointing\..*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Attribute 'logging_metrics' is an instance of `nn\.Module` and is already saved during checkpointing\..*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"The '.*_dataloader' does not have many workers which may be a bottleneck\..*",
        category=UserWarning,
    )


@contextmanager
def _quiet_lightning_info_logs() -> Iterator[None]:
    """
    Temporarily reduce noisy Lightning info logs during inference.
    """
    logger_names = (
        "lightning",
        "lightning.pytorch",
        "lightning.fabric",
        "lightning.pytorch.utilities.rank_zero",
    )
    previous_levels: Dict[str, int] = {}
    for name in logger_names:
        target = logging.getLogger(name)
        previous_levels[name] = target.level
        target.setLevel(logging.WARNING)
    try:
        yield
    finally:
        for name, level in previous_levels.items():
            logging.getLogger(name).setLevel(level)


def _select_checkpoint(model_path: Path) -> Path:
    """
    Prefer the best validation checkpoint when available.
    """
    best_ckpts = sorted(
        model_path.glob("best-*.ckpt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if best_ckpts:
        return best_ckpts[0]

    last_ckpt = model_path / "last.ckpt"
    if last_ckpt.exists():
        return last_ckpt

    ckpt_files = sorted(
        model_path.glob("*.ckpt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint found in {model_path}")
    return ckpt_files[0]


def _metric_to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)
    except Exception:
        return default


def _validate_forecast_array(arr: np.ndarray) -> bool:
    if arr is None:
        return False
    try:
        np_arr = np.asarray(arr, dtype=float)
    except Exception:
        return False
    if np_arr.size == 0:
        return False
    if np.isnan(np_arr).all():
        return False
    if np.isinf(np_arr).any():
        return False
    return True


def _prepare_dataframe(
    df: pd.DataFrame,
    prediction_length: Optional[int] = None,
) -> pd.DataFrame:
    """Prepare raw DataFrame by adding time_idx, target, and fixing dtypes."""
    cfg = settings.model
    pred_len = int(prediction_length or cfg.prediction_length)

    required = {"pair", "timestamp", "close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for TFT training: {sorted(missing)}")

    working = df.copy()
    working.sort_values(["pair", "timestamp"], inplace=True)
    working.reset_index(drop=True, inplace=True)
    working["timestamp"] = pd.to_datetime(working["timestamp"], errors="coerce")
    working = working.dropna(subset=["timestamp", "close", "pair"]).copy()

    working["time_idx"] = working.groupby("pair").cumcount()
    working["target"] = working.groupby("pair")["close"].transform(
        lambda x: np.log(x.shift(-pred_len) / x)
    )
    working.dropna(subset=["target"], inplace=True)

    cat_cols = [c for c in get_categorical_columns() if c in working.columns]
    for col in cat_cols:
        working[col] = working[col].astype(str)

    return working


def _filter_min_history(
    df: pd.DataFrame,
    encoder_length: int,
    prediction_length: int,
    required_pairs: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Drop pairs that do not have enough history for encoder+prediction windows."""
    min_rows = encoder_length + prediction_length + 5
    pair_sizes = df.groupby("pair").size()
    valid_pairs = pair_sizes[pair_sizes >= min_rows].index.tolist()
    filtered = df[df["pair"].isin(valid_pairs)].copy()
    if filtered.empty:
        raise RuntimeError(
            f"Not enough rows for TFT dataset. Need >= {min_rows} rows per pair after preprocessing."
        )
    if required_pairs:
        required_set = {str(pair).strip() for pair in required_pairs if str(pair).strip()}
        available = {str(pair).strip() for pair in filtered["pair"].unique()}
        missing_required = sorted(required_set.difference(available))
        if missing_required:
            counts = {
                str(pair): int(count)
                for pair, count in pair_sizes.sort_values(ascending=False).to_dict().items()
            }
            raise RuntimeError(
                "Required pair(s) missing after preprocessing/min-history filter: "
                f"{missing_required}. Pair row counts before filter: {counts}"
            )
    return filtered


def _resolve_num_workers() -> int:
    """Windows-safe default for DataLoader workers."""
    env_override = os.getenv("DATALOADER_NUM_WORKERS", "").strip()
    if env_override:
        try:
            return max(0, int(env_override))
        except ValueError:
            pass

    if os.name == "nt":
        return 0

    cpu_count = os.cpu_count() or 1
    return max(1, min(8, cpu_count - 1))


def _resolve_trainer_runtime() -> tuple[str, int]:
    """
    Resolve Trainer runtime accelerator safely.
    Falls back to CPU if CUDA is unavailable or unusable.
    """
    force_cpu = os.getenv("FORCE_CPU", "").strip().lower() in {"1", "true", "yes"}
    if force_cpu or not HAS_TORCH:
        return "cpu", 1

    if torch.cuda.is_available():
        try:
            _ = torch.zeros(1, device="cuda")
            return "gpu", 1
        except Exception as exc:
            logger.warning(f"CUDA appears unavailable at runtime, falling back to CPU: {exc}")

    return "cpu", 1


if HAS_TFT:
    _configure_lightning_warning_filters()


def _coerce_categories_to_encoder_vocab(
    df: pd.DataFrame,
    encoders: Dict[str, Any],
    pair_hint: Optional[str] = None,
) -> pd.DataFrame:
    """
    Replace out-of-vocabulary categorical values with known encoder categories.
    """
    sanitized = df.copy()
    for col_name, encoder in encoders.items():
        if col_name.startswith("__group_id__"):
            continue

        classes = getattr(encoder, "classes_", None)
        if not isinstance(classes, dict) or not classes:
            continue

        allowed = {str(key) for key in classes.keys()}
        fallback = str(next(iter(classes.keys())))
        if col_name == "pair" and pair_hint and pair_hint in allowed:
            fallback = pair_hint

        if col_name not in sanitized.columns:
            sanitized[col_name] = fallback
            continue

        series = sanitized[col_name].astype(str)
        sanitized[col_name] = np.where(series.isin(allowed), series, fallback)

    return sanitized


def prepare_tft_dataset(
    df: pd.DataFrame,
    encoder_length: Optional[int] = None,
    prediction_length: Optional[int] = None,
    training: bool = True,
    processed_df: Optional[pd.DataFrame] = None,
    required_pairs: Optional[List[str]] = None,
) -> "TimeSeriesDataSet":
    """
    Convert feature DataFrame into a TimeSeriesDataSet for TFT.
    """
    if not HAS_TFT:
        raise RuntimeError("pytorch-forecasting is required for TFT dataset creation")

    cfg = settings.model
    enc_len = int(encoder_length or cfg.encoder_length)
    pred_len = int(prediction_length or cfg.prediction_length)

    working = processed_df.copy() if processed_df is not None else _prepare_dataframe(df, pred_len)
    working = _filter_min_history(
        working,
        enc_len,
        pred_len,
        required_pairs=required_pairs if training else None,
    )

    cat_cols = [c for c in get_categorical_columns() if c in working.columns]
    cont_features = [c for c in get_feature_columns() if c in working.columns]
    known_reals = [c for c in ["hour_sin", "hour_cos", "dow_sin", "dow_cos"] if c in working.columns]
    unknown_reals = [c for c in cont_features if c not in known_reals]

    dataset_source = working
    if training:
        max_time_idx = int(working["time_idx"].max())
        training_cutoff = max(enc_len + pred_len, int(max_time_idx * 0.8))
        dataset_source = working[working["time_idx"] <= training_cutoff].copy()
        if dataset_source.empty:
            dataset_source = working.copy()

    dataset = TimeSeriesDataSet(
        dataset_source,
        time_idx="time_idx",
        target="target",
        group_ids=["pair"],
        min_encoder_length=max(4, enc_len // 2),
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


def build_tft_model(dataset: "TimeSeriesDataSet") -> "TemporalFusionTransformer":
    """Build a TFT model from a dataset."""
    if not HAS_TFT:
        raise RuntimeError("pytorch-forecasting required")

    cfg = settings.model
    return TemporalFusionTransformer.from_dataset(
        dataset,
        learning_rate=float(cfg.learning_rate),
        hidden_size=int(cfg.hidden_size),
        attention_head_size=int(cfg.attention_head_size),
        dropout=float(cfg.dropout),
        hidden_continuous_size=max(8, int(cfg.hidden_size) // 2),
        output_size=7,
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=5,
    )


def train_tft(
    df: pd.DataFrame,
    model_name: Optional[str] = None,
    max_epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    required_pairs: Optional[List[str]] = None,
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
    effective_epochs = int(max_epochs or cfg.max_epochs)
    effective_batch_size = int(batch_size or cfg.batch_size)
    version = model_name or f"tft_v{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    save_path = MODEL_DIR / version
    save_path.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Training TFT model: {version} | epochs={effective_epochs} | batch_size={effective_batch_size}"
    )

    required_pair_list = required_pairs or [XRP_ONLY_SYMBOL]
    required_pair_set = {str(pair).strip() for pair in required_pair_list if str(pair).strip()}
    if not required_pair_set:
        raise ValueError("required_pairs must contain at least one non-empty symbol")

    raw_pairs = {str(pair).strip() for pair in df.get("pair", pd.Series(dtype=str)).dropna().unique()}
    missing_raw = sorted(required_pair_set.difference(raw_pairs))
    if missing_raw:
        raise RuntimeError(
            "Training dataframe missing required pair(s): "
            f"{missing_raw}. Available pairs: {sorted(raw_pairs)}"
        )

    processed_df = _prepare_dataframe(df, cfg.prediction_length)
    processed_df = _filter_min_history(
        processed_df,
        encoder_length=int(cfg.encoder_length),
        prediction_length=int(cfg.prediction_length),
        required_pairs=sorted(required_pair_set),
    )

    training_dataset = prepare_tft_dataset(
        df=processed_df,
        training=True,
        processed_df=processed_df,
        encoder_length=int(cfg.encoder_length),
        prediction_length=int(cfg.prediction_length),
        required_pairs=sorted(required_pair_set),
    )

    val_source = processed_df.copy()
    dataset_encoders = getattr(training_dataset, "categorical_encoders", None)
    if isinstance(dataset_encoders, dict):
        val_source = _coerce_categories_to_encoder_vocab(val_source, dataset_encoders)

    val_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        val_source,
        predict=False,
        stop_randomization=True,
    )

    workers = _resolve_num_workers()
    train_loader = training_dataset.to_dataloader(
        train=True,
        batch_size=effective_batch_size,
        num_workers=workers,
    )
    val_loader = val_dataset.to_dataloader(
        train=False,
        batch_size=max(1, effective_batch_size * 2),
        num_workers=workers,
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=max(3, min(10, effective_epochs)),
        min_delta=1e-5,
        mode="min",
    )
    checkpoint = ModelCheckpoint(
        dirpath=str(save_path),
        filename="best-{epoch}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    accelerator, devices = _resolve_trainer_runtime()
    with _quiet_lightning_info_logs():
        trainer = pl.Trainer(
            max_epochs=effective_epochs,
            accelerator=accelerator,
            devices=devices,
            logger=False,
            gradient_clip_val=0.1,
            callbacks=[early_stop, checkpoint],
            enable_progress_bar=True,
            log_every_n_steps=10,
            enable_model_summary=False,
        )

        model = build_tft_model(training_dataset)
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        best_model_path = checkpoint.best_model_path
        if not best_model_path:
            fallback_ckpt = save_path / "last.ckpt"
            trainer.save_checkpoint(str(fallback_ckpt))
            best_model_path = str(fallback_ckpt)

        best_model = TemporalFusionTransformer.load_from_checkpoint(
            best_model_path,
            map_location="cpu",
        )

        val_metrics = trainer.validate(best_model, dataloaders=val_loader)
    metrics = {
        "training_loss": _metric_to_float(trainer.callback_metrics.get("train_loss")),
        "validation_loss": _metric_to_float(val_metrics[0].get("val_loss")) if val_metrics else 0.0,
        "best_epoch": float(early_stop.stopped_epoch or effective_epochs),
    }

    info_path = save_path / "info.txt"
    with open(info_path, "w", encoding="utf-8") as handle:
        handle.write(f"version: {version}\n")
        handle.write(f"trained_at: {datetime.utcnow().isoformat()}\n")
        handle.write(f"metrics: {metrics}\n")
        handle.write(f"accelerator: {accelerator}\n")

    logger.info(f"Model {version} trained. Val loss: {metrics['validation_loss']:.6f}")
    return version, metrics


class _PredictorLoadDescriptor:
    """Support both `TFTPredictor.load(version)` and `predictor.load(version)`."""

    def __get__(
        self,
        instance: Optional["TFTPredictor"],
        owner: type["TFTPredictor"],
    ) -> Any:
        if instance is None:

            def _load_from_class(model_version: str) -> "TFTPredictor":
                predictor = owner()
                predictor._load_model(model_version)
                return predictor

            return _load_from_class

        def _load_from_instance(model_version: str) -> "TFTPredictor":
            instance._load_model(model_version)
            return instance

        return _load_from_instance


class TFTPredictor:
    """Thin predictor wrapper: model loading, version tracking, stable predict interface."""

    load = _PredictorLoadDescriptor()

    def __init__(self, model_version: Optional[str] = None) -> None:
        self.model: Optional[TemporalFusionTransformer] = None
        self.model_version: str = ""
        self.device: str = "cpu"
        if model_version:
            self._load_model(model_version)

    def _load_model(self, model_version: str) -> None:
        """Load a saved TFT checkpoint into memory."""
        if not HAS_TFT:
            raise RuntimeError("pytorch-forecasting required")

        version = str(model_version).strip()
        if not version:
            raise ValueError("model_version must be a non-empty string")

        model_path = MODEL_DIR / version
        best_ckpt = _select_checkpoint(model_path)

        loaded_model = TemporalFusionTransformer.load_from_checkpoint(
            str(best_ckpt),
            map_location="cpu",
        )
        if loaded_model is None:
            raise RuntimeError(f"Failed to load checkpoint: {best_ckpt}")

        target_device = "cpu"
        if HAS_TORCH and torch.cuda.is_available():
            try:
                _ = torch.zeros(1, device="cuda")
                target_device = "cuda"
            except Exception as exc:
                logger.warning(f"CUDA unavailable for inference, using CPU: {exc}")

        loaded_model.to(target_device)
        loaded_model.eval()

        self.model = loaded_model
        self.model_version = version
        self.device = target_device
        logger.info(f"Loaded model: {version} from {best_ckpt} on {target_device}")

    def _empty_prediction(self) -> Dict[str, Any]:
        return {
            "prob_up": 0.5,
            "prob_down": 0.5,
            "expected_move": 0.0,
            "confidence": 0.0,
            "valid": False,
            "forecast_vector": [],
            "lower_bound": [],
            "upper_bound": [],
            "attention_stats": {
                "mean": 0.0,
                "std": 0.0,
                "peak": 0.0,
                "consistency": 0.0,
            },
            "model_version": self.model_version or "",
        }

    @staticmethod
    def _extract_allowed_classes(encoder: Any) -> List[str]:
        classes = getattr(encoder, "classes_", None)
        if classes is None:
            return []

        raw_values: List[Any]
        if isinstance(classes, dict):
            raw_values = list(classes.keys())
        elif hasattr(classes, "tolist"):
            try:
                raw = classes.tolist()
            except Exception:
                raw = classes
            raw_values = raw if isinstance(raw, list) else list(raw) if isinstance(raw, tuple) else [raw]
        elif isinstance(classes, (list, tuple, set)):
            raw_values = list(classes)
        else:
            return []

        out: List[str] = []
        seen: set[str] = set()
        for value in raw_values:
            text = str(value).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            out.append(text)
        return out

    def get_supported_pairs(self) -> List[str]:
        """
        Return the trained pair vocabulary from model categorical encoders.
        """
        if self.model is None:
            return []

        dataset_params = getattr(self.model, "dataset_parameters", None)
        if not isinstance(dataset_params, dict):
            return []

        encoders = dataset_params.get("categorical_encoders")
        if not isinstance(encoders, dict):
            return []

        pair_values: List[str] = []
        for key in ("pair", "__group_id__pair"):
            if key in encoders:
                pair_values.extend(self._extract_allowed_classes(encoders[key]))

        if not pair_values:
            for name, encoder in encoders.items():
                if "pair" in str(name).lower():
                    pair_values.extend(self._extract_allowed_classes(encoder))

        unique = sorted({value for value in pair_values if value})
        return unique

    def _build_inference_dataset(self, df: pd.DataFrame, pair: str) -> "TimeSeriesDataSet":
        """
        Build inference dataset aligned to trained model encoders.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        processed = _prepare_dataframe(df, settings.model.prediction_length)
        processed = _filter_min_history(
            processed,
            encoder_length=int(settings.model.encoder_length),
            prediction_length=int(settings.model.prediction_length),
        )

        dataset_params = getattr(self.model, "dataset_parameters", None)
        if isinstance(dataset_params, dict):
            processed = self._coerce_known_categories(processed, dataset_params, pair)
            try:
                return TimeSeriesDataSet.from_parameters(
                    dataset_params,
                    processed,
                    predict=True,
                    stop_randomization=True,
                )
            except Exception as exc:
                raise RuntimeError(f"from_parameters dataset build failed for {pair}: {exc}") from exc

        return prepare_tft_dataset(df=processed, training=False)

    def _coerce_known_categories(
        self,
        df: pd.DataFrame,
        dataset_params: Dict[str, Any],
        pair: str,
    ) -> pd.DataFrame:
        """
        Replace unseen categorical values with known encoder-safe fallbacks.
        """
        encoders = dataset_params.get("categorical_encoders")
        if not isinstance(encoders, dict):
            return df

        sanitized = df.copy()
        pair_vocab: Optional[set[str]] = None

        for col_name, encoder in encoders.items():
            if col_name.startswith("__group_id__"):
                continue

            allowed_values = self._extract_allowed_classes(encoder)
            if not allowed_values:
                continue

            allowed = set(allowed_values)
            if col_name == "pair":
                pair_vocab = allowed

            fallback = allowed_values[0]
            if col_name == "pair" and pair in allowed:
                fallback = pair

            if col_name not in sanitized.columns:
                sanitized[col_name] = fallback
                continue

            series = sanitized[col_name].astype(str)
            sanitized[col_name] = np.where(series.isin(allowed), series, fallback)

        if pair_vocab is not None and pair not in pair_vocab:
            raise ValueError(
                f"Pair '{pair}' is not in model vocabulary: {sorted(pair_vocab)}"
            )

        return sanitized

    def predict(self, df: pd.DataFrame, pair: str) -> Dict[str, Any]:
        """
        Run inference for one pair.
        Never raises to caller; returns `_empty_prediction()` on failure.
        """
        logger.info(
            f"TFTPredictor.predict called | pair={pair} | rows={len(df) if isinstance(df, pd.DataFrame) else -1} "
            f"| model_version={self.model_version or 'unloaded'}"
        )

        if self.model is None:
            logger.warning("Predict called with no loaded model.")
            return self._empty_prediction()

        if not isinstance(df, pd.DataFrame) or df.empty:
            logger.warning("Predict called with empty or invalid dataframe.")
            return self._empty_prediction()

        if "pair" not in df.columns:
            logger.warning("Predict input dataframe missing 'pair' column.")
            return self._empty_prediction()

        if not hasattr(self.model, "predict"):
            logger.error("Loaded model does not expose a predict() method.")
            return self._empty_prediction()

        working = df[df["pair"] == pair].copy()
        required_len = settings.model.encoder_length + settings.model.prediction_length + 5
        if len(working) < required_len:
            logger.warning(
                f"Insufficient history for {pair}: {len(working)} < required {required_len}"
            )
            return self._empty_prediction()

        try:
            dataset = self._build_inference_dataset(working, pair)
            if dataset is None or len(dataset) == 0:
                logger.warning(f"TFT dataset produced zero windows for {pair}")
                return self._empty_prediction()

            dataloader = dataset.to_dataloader(
                train=False,
                batch_size=1,
                num_workers=_resolve_num_workers(),
            )

            trainer_kwargs = {
                "logger": False,
                "enable_checkpointing": False,
                "enable_progress_bar": False,
                "enable_model_summary": False,
            }
            with _quiet_lightning_info_logs():
                raw_preds = self.model.predict(
                    dataloader,
                    mode="quantiles",
                    return_x=False,
                    trainer_kwargs=trainer_kwargs,
                )

            if raw_preds is None or len(raw_preds) == 0:
                logger.warning(f"Model returned empty predictions for {pair}")
                return self._empty_prediction()

            raw_np = raw_preds.detach().cpu().numpy()
            if raw_np.ndim < 2 or raw_np.shape[0] == 0 or not _validate_forecast_array(raw_np):
                logger.warning(f"Invalid prediction tensor shape for {pair}: {raw_np.shape}")
                return self._empty_prediction()

            last_pred = raw_np[-1]
            if not _validate_forecast_array(last_pred):
                logger.warning(f"Invalid forecast values for {pair}; using fallback prediction")
                return self._empty_prediction()

            if raw_np.ndim == 3 and raw_np.shape[2] >= 7:
                median_forecast = np.asarray(last_pred[:, 3], dtype=float)
                lower_bound = np.asarray(last_pred[:, 0], dtype=float)
                upper_bound = np.asarray(last_pred[:, 6], dtype=float)

                if not (
                    _validate_forecast_array(median_forecast)
                    and _validate_forecast_array(lower_bound)
                    and _validate_forecast_array(upper_bound)
                ):
                    logger.warning(f"Quantile forecast invalid for {pair}; using fallback prediction")
                    return self._empty_prediction()

                positive_quantiles = sum(
                    1 for i in range(7)
                    if float(np.nanmean(last_pred[:, i])) > 0
                )
                prob_up = positive_quantiles / 7.0
            else:
                median_forecast = np.asarray(last_pred, dtype=float)
                if not _validate_forecast_array(median_forecast):
                    logger.warning(f"Point forecast invalid for {pair}; using fallback prediction")
                    return self._empty_prediction()

                horizon_std = float(np.nanstd(median_forecast))
                lower_bound = median_forecast - 2 * horizon_std
                upper_bound = median_forecast + 2 * horizon_std

                valid_mask = ~np.isnan(raw_np).any(axis=1)
                if valid_mask.sum() > 0:
                    valid_means = np.nanmean(raw_np[valid_mask], axis=1)
                    prob_up = float((valid_means > 0).mean())
                else:
                    prob_up = 0.5

            if not np.isfinite(prob_up):
                logger.warning(f"Non-finite probability for {pair}; using fallback prediction")
                return self._empty_prediction()

            prob_down = 1.0 - prob_up
            expected_move = float(np.nanmean(median_forecast))
            if not np.isfinite(expected_move):
                logger.warning(f"Non-finite expected move for {pair}; using fallback prediction")
                return self._empty_prediction()

            spread = float(np.nanmean(upper_bound - lower_bound))
            mean_abs_forecast = float(np.nanmean(np.abs(median_forecast))) + 1e-8
            if not np.isfinite(spread) or not np.isfinite(mean_abs_forecast):
                logger.warning(f"Non-finite forecast spread stats for {pair}; using fallback prediction")
                return self._empty_prediction()

            raw_uncertainty = spread / mean_abs_forecast
            tightness = max(0.0, min(1.0, 1.0 / (1.0 + raw_uncertainty * 0.5)))
            direction_score = max(prob_up, prob_down)

            if len(median_forecast) > 1:
                sign_changes = np.sum(np.diff(np.sign(median_forecast)) != 0)
                consistency = 1.0 - sign_changes / (len(median_forecast) - 1)
            else:
                consistency = 0.5

            attention_stats = {
                "mean": float(np.nanmean(np.abs(median_forecast))),
                "std": float(np.nanstd(median_forecast)),
                "peak": float(np.nanmax(np.abs(median_forecast))) if len(median_forecast) > 0 else 0.0,
                "consistency": float(consistency),
            }

            confidence = (
                0.30 * tightness
                + 0.40 * direction_score
                + 0.30 * consistency
            )
            confidence = round(max(0.1, min(1.0, confidence)), 4)

            if mean_abs_forecast < 1e-6:
                confidence = min(confidence, 0.35)

            return {
                "prob_up": round(prob_up, 4),
                "prob_down": round(prob_down, 4),
                "expected_move": round(expected_move, 6),
                "confidence": confidence,
                "valid": True,
                "forecast_vector": median_forecast.tolist(),
                "lower_bound": lower_bound.tolist(),
                "upper_bound": upper_bound.tolist(),
                "attention_stats": attention_stats,
                "model_version": self.model_version or "",
            }
        except Exception as exc:
            logger.error(f"Prediction error for {pair}: {exc}")
            return self._empty_prediction()

    @staticmethod
    def list_models() -> List[str]:
        """List all saved model versions."""
        if not MODEL_DIR.exists():
            return []
        return sorted([d.name for d in MODEL_DIR.iterdir() if d.is_dir()])
