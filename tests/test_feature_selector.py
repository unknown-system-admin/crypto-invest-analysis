import pandas as pd
import numpy as np
from analyzers.feature_selector import FeatureSelector


def test_feature_selector_train():
    dates = pd.date_range("2024-01-01", periods=300, freq="1D")
    np.random.seed(42)
    
    df = pd.DataFrame({
        "close": 50000 + np.cumsum(np.random.randn(300) * 500),
        "RSI": np.random.uniform(30, 70, 300),
        "MACD": np.random.randn(300) * 100,
        "SMA_20": np.random.uniform(40000, 50000, 300),
        "SMA_50": np.random.uniform(40000, 50000, 300),
    }, index=dates)
    
    # Add binary label
    df["binary_label"] = (df["close"].shift(-5) > df["close"]).astype(int)
    df = df.dropna()
    
    selector = FeatureSelector()
    result = selector.train(
        df,
        feature_columns=["RSI", "MACD", "SMA_20", "SMA_50"],
        label_column="binary_label",
    )
    
    assert "accuracy" in result
    assert "feature_importance" in result
    assert len(result["feature_importance"]) == 4


def test_feature_selector_get_top_features():
    selector = FeatureSelector()
    selector.feature_importance = {
        "RSI": 0.3,
        "MACD": 0.25,
        "SMA_20": 0.2,
        "SMA_50": 0.15,
        "ATR": 0.1,
    }
    
    top = selector.get_top_features(n=3)
    
    assert len(top) == 3
    assert top[0] == "RSI"
    assert top[1] == "MACD"
