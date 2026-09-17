# feature_engine/builder.py
import pandas as pd
from feature_engine.indicators import compute_all_indicators
from feature_engine.momentum import momentum_score, momentum_delta, momentum_acceleration
from feature_engine.labels import binary_label


def _attach_momentum(df: pd.DataFrame) -> pd.DataFrame:
    indicators = compute_all_indicators(df)
    indicators["close"] = df["close"]
    momentum = momentum_score(indicators)
    indicators["momentum_score"] = momentum
    indicators["momentum_delta"] = momentum_delta(momentum)
    indicators["momentum_acceleration"] = momentum_acceleration(indicators["momentum_delta"])
    return indicators


def build_live_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features for live /signal evaluation.

    Does NOT compute future-return labels, so the latest bars are kept.
    Only drops indicator warmup NaNs at the start of the series.
    """
    features = _attach_momentum(df)
    return features.dropna()


def build_feature_matrix(df: pd.DataFrame, n_bars: int = 5, include_labels: bool = True):
    """Build feature matrix.

    Training default (include_labels=True) still drops the last n_bars so
    labels have no lookahead leakage.

    Live path should use include_labels=False or build_live_features().
    """
    features = _attach_momentum(df)

    if not include_labels:
        features = features.dropna()
        return features, None

    labels = binary_label(df, n_bars=n_bars)
    valid_idx = features.dropna().index.intersection(labels.dropna().index)
    features = features.loc[valid_idx]
    labels = labels.loc[valid_idx]
    return features, labels
