import pandas as pd
import numpy as np


def momentum_score(df: pd.DataFrame) -> pd.Series:
    """Calculate momentum score as continuous value [-1, 1].

    Args:
        df: DataFrame with columns: close, RSI, MACD_histogram, SMA_20, SMA_50

    Returns:
        Series with momentum scores in range [-1, 1]
    """
    rsi_norm = (df["RSI"] - 50) / 50
    macd_norm = np.tanh(df["MACD_histogram"] / df["MACD_histogram"].std())
    sma20_norm = (df["close"] - df["SMA_20"]) / df["SMA_20"]
    sma50_norm = (df["close"] - df["SMA_50"]) / df["SMA_50"]

    score = (
        0.3 * rsi_norm +
        0.3 * macd_norm +
        0.2 * sma20_norm +
        0.2 * sma50_norm
    )

    score = score.clip(-1, 1)

    return score


def momentum_delta(scores: pd.Series) -> pd.Series:
    """Calculate first derivative of momentum score.

    Args:
        scores: Momentum scores

    Returns:
        Series with delta values
    """
    return scores.diff()


def momentum_acceleration(deltas: pd.Series) -> pd.Series:
    """Calculate second derivative of momentum score.

    Args:
        deltas: Delta values from momentum_delta

    Returns:
        Series with acceleration values
    """
    return deltas.diff()
