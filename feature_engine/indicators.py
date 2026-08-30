import pandas as pd
import ta


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute optimized technical indicators (9 features).

    Args:
        df: OHLCV DataFrame with columns: open, high, low, close, volume

    Returns:
        DataFrame with optimized indicator columns
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # Trend indicators (kept)
    sma_20 = ta.trend.sma_indicator(close, window=20)
    sma_50 = ta.trend.sma_indicator(close, window=50)
    sma_200 = ta.trend.sma_indicator(close, window=200)
    ema_26 = ta.trend.ema_indicator(close, window=26)

    # Volatility indicators (kept)
    atr = ta.volatility.average_true_range(high, low, close, window=14)

    # Momentum indicators (kept)
    rsi = ta.momentum.rsi(close, window=14)
    macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    macd_line = macd.macd()

    # Volume indicators (kept)
    mfi = ta.volume.money_flow_index(high, low, close, volume, window=14)
    obv = ta.volume.on_balance_volume(close, volume)

    result = pd.DataFrame({
        "SMA_20": sma_20,
        "SMA_50": sma_50,
        "SMA_200": sma_200,
        "EMA_26": ema_26,
        "ATR": atr,
        "RSI": rsi,
        "MACD": macd_line,
        "MFI": mfi,
        "OBV": obv,
    }, index=df.index)

    return result
