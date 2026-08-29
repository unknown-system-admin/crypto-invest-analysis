import pandas as pd
import ta


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all 18 technical indicators.

    Args:
        df: OHLCV DataFrame with columns: open, high, low, close, volume

    Returns:
        DataFrame with all indicator columns
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # Trend indicators
    sma_20 = ta.trend.sma_indicator(close, window=20)
    sma_50 = ta.trend.sma_indicator(close, window=50)
    sma_200 = ta.trend.sma_indicator(close, window=200)
    ema_12 = ta.trend.ema_indicator(close, window=12)
    ema_26 = ta.trend.ema_indicator(close, window=26)
    adx = ta.trend.adx(high, low, close, window=14)

    # Ichimoku Cloud
    ichimoku = ta.trend.IchimokuIndicator(high, low, window1=9, window2=26, window3=52)
    ichimoku_a = ichimoku.ichimoku_a()
    ichimoku_b = ichimoku.ichimoku_b()

    # Volatility indicators
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    bb_upper = bb.bollinger_hband()
    bb_middle = bb.bollinger_mavg()
    bb_lower = bb.bollinger_lband()
    atr = ta.volatility.average_true_range(high, low, close, window=14)

    # Keltner Channels
    kc = ta.volatility.KeltnerChannel(high, low, close, window=20, window_atr=20)
    kc_upper = kc.keltner_channel_hband()
    kc_lower = kc.keltner_channel_lband()

    # Momentum indicators
    rsi = ta.momentum.rsi(close, window=14)
    macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    macd_line = macd.macd()
    macd_signal = macd.macd_signal()
    macd_histogram = macd.macd_diff()

    stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
    stoch_k = stoch.stoch()
    stoch_d = stoch.stoch_signal()

    cci = ta.trend.cci(high, low, close, window=20)
    williams_r = ta.momentum.williams_r(high, low, close, lbp=14)
    roc = ta.momentum.roc(close, window=12)
    mfi = ta.volume.money_flow_index(high, low, close, volume, window=14)

    # Volume indicators
    obv = ta.volume.on_balance_volume(close, volume)
    vwap = (volume * (high + low + close) / 3).cumsum() / volume.cumsum()
    cmf = ta.volume.chaikin_money_flow(high, low, close, volume, window=20)

    result = pd.DataFrame({
        "SMA_20": sma_20,
        "SMA_50": sma_50,
        "SMA_200": sma_200,
        "EMA_12": ema_12,
        "EMA_26": ema_26,
        "ADX": adx,
        "ICHIMOKU_A": ichimoku_a,
        "ICHIMOKU_B": ichimoku_b,
        "BB_upper": bb_upper,
        "BB_middle": bb_middle,
        "BB_lower": bb_lower,
        "ATR": atr,
        "KC_upper": kc_upper,
        "KC_lower": kc_lower,
        "RSI": rsi,
        "MACD": macd_line,
        "MACD_signal": macd_signal,
        "MACD_histogram": macd_histogram,
        "STOCH_K": stoch_k,
        "STOCH_D": stoch_d,
        "CCI": cci,
        "Williams_R": williams_r,
        "ROC": roc,
        "MFI": mfi,
        "OBV": obv,
        "VWAP": vwap,
        "CMF": cmf,
    }, index=df.index)

    return result
