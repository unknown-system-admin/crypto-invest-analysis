import pandas as pd
import numpy as np
from feature_engine.indicators import compute_all_indicators


def test_compute_all_indicators_returns_expected_columns():
    dates = pd.date_range("2024-01-01", periods=100, freq="1h")
    df = pd.DataFrame({
        "open": np.random.uniform(40000, 50000, 100),
        "high": np.random.uniform(40000, 50000, 100),
        "low": np.random.uniform(40000, 50000, 100),
        "close": np.random.uniform(40000, 50000, 100),
        "volume": np.random.uniform(100, 1000, 100),
    }, index=dates)

    result = compute_all_indicators(df)

    expected_columns = [
        "SMA_20", "SMA_50", "SMA_200", "EMA_12", "EMA_26",
        "ADX", "ICHIMOKU_A", "ICHIMOKU_B",
        "BB_upper", "BB_middle", "BB_lower", "ATR", "KC_upper", "KC_lower",
        "RSI", "MACD", "MACD_signal", "MACD_histogram",
        "STOCH_K", "STOCH_D", "CCI", "Williams_R", "ROC", "MFI",
        "OBV", "VWAP", "CMF",
    ]
    for col in expected_columns:
        assert col in result.columns, f"Missing column: {col}"


def test_compute_all_indicators_handles_nan_gracefully():
    dates = pd.date_range("2024-01-01", periods=50, freq="1h")
    df = pd.DataFrame({
        "open": np.random.uniform(40000, 50000, 50),
        "high": np.random.uniform(40000, 50000, 50),
        "low": np.random.uniform(40000, 50000, 50),
        "close": np.random.uniform(40000, 50000, 50),
        "volume": np.random.uniform(100, 1000, 50),
    }, index=dates)

    result = compute_all_indicators(df)
    assert not result.empty
