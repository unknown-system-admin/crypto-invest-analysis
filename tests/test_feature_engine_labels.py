# tests/test_feature_engine_labels.py
import pandas as pd
import numpy as np
from feature_engine.labels import future_return, binary_label


def test_future_return_positive():
    dates = pd.date_range("2024-01-01", periods=10, freq="1h")
    df = pd.DataFrame({
        "close": [100, 105, 110, 115, 120, 125, 130, 135, 140, 145],
    }, index=dates)
    
    result = future_return(df, n_bars=2)
    
    assert result.iloc[0] == 10.0  # (110 - 100) / 100 * 100
    assert result.iloc[7] == 10 / 135 * 100  # (145 - 135) / 135 * 100


def test_future_return_negative():
    dates = pd.date_range("2024-01-01", periods=10, freq="1h")
    df = pd.DataFrame({
        "close": [140, 135, 130, 125, 120, 115, 110, 105, 100, 95],
    }, index=dates)
    
    result = future_return(df, n_bars=2)
    
    assert result.iloc[0] < 0


def test_binary_label_up():
    dates = pd.date_range("2024-01-01", periods=10, freq="1h")
    df = pd.DataFrame({
        "close": [100, 105, 110, 115, 120, 125, 130, 135, 140, 145],
    }, index=dates)
    
    result = binary_label(df, n_bars=2, threshold=0)
    
    assert result.iloc[0] == 1  # Up


def test_binary_label_down():
    dates = pd.date_range("2024-01-01", periods=10, freq="1h")
    df = pd.DataFrame({
        "close": [140, 135, 130, 125, 120, 115, 110, 105, 100, 95],
    }, index=dates)
    
    result = binary_label(df, n_bars=2, threshold=0)
    
    assert result.iloc[0] == 0  # Down
