import pandas as pd
import ta


def compute_all(df: pd.DataFrame) -> dict:
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    sma_20 = ta.trend.sma_indicator(close, window=20)
    sma_50 = ta.trend.sma_indicator(close, window=50)
    sma_200 = ta.trend.sma_indicator(close, window=200)
    ema_12 = ta.trend.ema_indicator(close, window=12)
    ema_26 = ta.trend.ema_indicator(close, window=26)
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)

    overlay = pd.DataFrame({
        "open": df["open"],
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "SMA_20": sma_20,
        "SMA_50": sma_50,
        "SMA_200": sma_200,
        "EMA_12": ema_12,
        "EMA_26": ema_26,
        "BB_upper": bb.bollinger_hband(),
        "BB_middle": bb.bollinger_mavg(),
        "BB_lower": bb.bollinger_lband(),
    }, index=df.index)

    rsi = ta.momentum.rsi(close, window=14)
    macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
    obv = ta.volume.on_balance_volume(close, volume)

    subplots = {
        "rsi": pd.DataFrame({"RSI": rsi}, index=df.index),
        "macd": pd.DataFrame({
            "MACD": macd.macd(),
            "Signal": macd.macd_signal(),
            "Histogram": macd.macd_diff(),
        }, index=df.index),
        "stoch": pd.DataFrame({
            "%K": stoch.stoch(),
            "%D": stoch.stoch_signal(),
        }, index=df.index),
        "obv": pd.DataFrame({"OBV": obv}, index=df.index),
    }

    return {"overlay": overlay, "subplots": subplots}
