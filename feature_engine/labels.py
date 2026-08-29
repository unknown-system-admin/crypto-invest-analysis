# feature_engine/labels.py
import pandas as pd


def future_return(df: pd.DataFrame, n_bars: int = 5) -> pd.Series:
    """Calculate future return percentage.
    
    Args:
        df: DataFrame with 'close' column
        n_bars: Number of bars to look ahead
        
    Returns:
        Series with future return percentages
    """
    future_prices = df["close"].shift(-n_bars)
    return ((future_prices - df["close"]) / df["close"]) * 100


def binary_label(df: pd.DataFrame, n_bars: int = 5, threshold: float = 0) -> pd.Series:
    """Generate binary labels (1=up, 0=down) based on future returns.
    
    Args:
        df: DataFrame with 'close' column
        n_bars: Number of bars to look ahead
        threshold: Minimum return percentage to label as up
        
    Returns:
        Series with binary labels (1 for up, 0 for down)
    """
    returns = future_return(df, n_bars)
    return (returns > threshold).astype(int)
