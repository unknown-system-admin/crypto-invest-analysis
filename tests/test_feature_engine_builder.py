# tests/test_feature_engine_builder.py
import pandas as pd
import numpy as np
from feature_engine.builder import build_feature_matrix


def test_build_feature_matrix_returns_correct_shape():
    # Use 300 rows because sma_200 requires 200 periods
    dates = pd.date_range("2024-01-01", periods=300, freq="1h")
    df = pd.DataFrame({
        "open": np.random.uniform(40000, 50000, 300),
        "high": np.random.uniform(40000, 50000, 300),
        "low": np.random.uniform(40000, 50000, 300),
        "close": np.random.uniform(40000, 50000, 300),
        "volume": np.random.uniform(100, 1000, 300),
    }, index=dates)
    
    features, labels = build_feature_matrix(df, n_bars=5)
    
    assert not features.empty
    assert not labels.empty
    assert len(features) == len(labels)


def test_build_feature_matrix_handles_nan():
    # Use 300 rows because sma_200 requires 200 periods
    dates = pd.date_range("2024-01-01", periods=300, freq="1h")
    df = pd.DataFrame({
        "open": np.random.uniform(40000, 50000, 300),
        "high": np.random.uniform(40000, 50000, 300),
        "low": np.random.uniform(40000, 50000, 300),
        "close": np.random.uniform(40000, 50000, 300),
        "volume": np.random.uniform(100, 1000, 300),
    }, index=dates)
    
    features, labels = build_feature_matrix(df, n_bars=5)
    
    assert features.isna().sum().sum() == 0
    assert labels.isna().sum() == 0