from data.database import init_db, get_session, Trade, Prediction, ModelMetric, LearningMetric, EngineState, DailyStats
from data.features import compute_features, align_multi_timeframe, get_feature_columns, get_categorical_columns

try:
    from data.fetcher import KuCoinDataFetcher
except Exception:  # pragma: no cover - optional import guard for environments without exchange deps
    KuCoinDataFetcher = None

__all__ = [
    "init_db", "get_session", "Trade", "Prediction", "ModelMetric", "LearningMetric",
    "EngineState", "DailyStats", "KuCoinDataFetcher",
    "compute_features", "align_multi_timeframe", "get_feature_columns", "get_categorical_columns",
]
