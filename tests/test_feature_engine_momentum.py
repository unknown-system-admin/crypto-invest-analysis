import pandas as pd
import numpy as np
from feature_engine.momentum import momentum_score, momentum_delta, momentum_acceleration


def test_momentum_score_range():
    dates = pd.date_range("2024-01-01", periods=100, freq="1h")
    df = pd.DataFrame({
        "close": np.random.uniform(40000, 50000, 100),
        "RSI": np.random.uniform(30, 70, 100),
        "MACD_histogram": np.random.uniform(-100, 100, 100),
        "SMA_20": np.random.uniform(40000, 50000, 100),
        "SMA_50": np.random.uniform(40000, 50000, 100),
    }, index=dates)

    scores = momentum_score(df)

    assert scores.min() >= -1.0
    assert scores.max() <= 1.0
    assert len(scores) == 100


def test_momentum_delta_positive_when_increasing():
    scores = pd.Series([0.1, 0.2, 0.4, 0.6, 0.8])
    deltas = momentum_delta(scores)

    assert deltas.iloc[1] > 0
    assert deltas.iloc[2] > 0
    assert deltas.iloc[3] > 0


def test_momentum_delta_negative_when_decreasing():
    scores = pd.Series([0.8, 0.6, 0.4, 0.2, 0.1])
    deltas = momentum_delta(scores)

    assert deltas.iloc[1] < 0
    assert deltas.iloc[2] < 0
    assert deltas.iloc[3] < 0


def test_momentum_acceleration_positive_when_accelerating():
    scores = pd.Series([0.1, 0.2, 0.4, 0.7, 1.1])
    deltas = momentum_delta(scores)
    accelerations = momentum_acceleration(deltas)

    assert accelerations.iloc[2] > 0


def test_momentum_acceleration_negative_when_decelerating():
    scores = pd.Series([0.1, 0.4, 0.5, 0.55, 0.575])
    deltas = momentum_delta(scores)
    accelerations = momentum_acceleration(deltas)

    assert accelerations.iloc[2] < 0
