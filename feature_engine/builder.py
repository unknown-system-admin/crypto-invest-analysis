# feature_engine/builder.py
import pandas as pd
from feature_engine.indicators import compute_all_indicators
from feature_engine.momentum import momentum_score, momentum_delta, momentum_acceleration
from feature_engine.labels import binary_label


def build_feature_matrix(df: pd.DataFrame, n_bars: int = 5) -> tuple:
    """Build complete feature matrix with labels.
    
    Args:
        df: OHLCV DataFrame
        n_bars: Number of bars to look ahead for labels
        
    Returns:
        Tuple of (features DataFrame, labels Series)
    """
    # Compute indicators
    indicators = compute_all_indicators(df)
    
    # Compute momentum (needs close column for normalization)
    indicators["close"] = df["close"]
    momentum = momentum_score(indicators)
    delta = momentum_delta(momentum)
    acceleration = momentum_acceleration(delta)
    
    # Add momentum features
    indicators["momentum_score"] = momentum
    indicators["momentum_delta"] = delta
    indicators["momentum_acceleration"] = acceleration
    
    # Generate labels
    labels = binary_label(df, n_bars=n_bars)
    
    # Combine features
    features = indicators.copy()
    
    # Drop rows with NaN (first 200 periods for indicators, last n_bars for labels)
    valid_idx = features.dropna().index.intersection(labels.dropna().index)
    features = features.loc[valid_idx]
    labels = labels.loc[valid_idx]
    
    return features, labels