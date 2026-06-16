import pandas as pd
import numpy as np
from trading.strategy import MACross, RSIThreshold, MACDCross, CompositeStrategy, CustomComposite, Signal


def _make_overlay(close_vals, sma_vals=None):
    df = pd.DataFrame({"close": close_vals})
    if sma_vals:
        df["SMA_20"] = sma_vals
        df["SMA_50"] = sma_vals
    df["EMA_12"] = close_vals
    df["EMA_26"] = close_vals
    df["SMA_200"] = close_vals
    return df


def test_ma_cross_golden():
    close = [100, 101, 102, 103, 104, 105]
    fast = [100, 101, 102, 102.5, 102, 104]
    slow = [103, 103, 103, 103, 103, 103]
    overlay = pd.DataFrame({"close": close, "EMA_12": fast, "EMA_26": slow,
                            "SMA_20": close, "SMA_50": close, "SMA_200": close})
    s = MACross(fast=12, slow=26, ma_type="ema")
    sig = s.evaluate(overlay, {})
    assert sig.direction == "偏多"


def test_ma_cross_death():
    close = [105, 104, 103, 102, 101, 100]
    fast = [105, 104, 104, 104, 104, 102]
    slow = [103, 103, 103, 103, 103, 103]
    overlay = pd.DataFrame({"close": close, "EMA_12": fast, "EMA_26": slow,
                            "SMA_20": close, "SMA_50": close, "SMA_200": close})
    s = MACross(fast=12, slow=26, ma_type="ema")
    sig = s.evaluate(overlay, {})
    assert sig.direction == "偏空"


def test_ma_cross_no_cross():
    close = [100, 101, 102, 103, 104, 105]
    fast = [101, 101, 101, 101, 101, 101]
    slow = [103, 103, 103, 103, 103, 103]
    overlay = pd.DataFrame({"close": close, "EMA_12": fast, "EMA_26": slow,
                            "SMA_20": close, "SMA_50": close, "SMA_200": close})
    s = MACross(fast=12, slow=26, ma_type="ema")
    sig = s.evaluate(overlay, {})
    assert sig.direction == "中立"


def test_rsi_oversold_bullish():
    rsi_vals = [28, 28, 28, 28, 28, 29]
    subplots = {"rsi": pd.DataFrame({"RSI": rsi_vals})}
    s = RSIThreshold(period=14, overbought=70, oversold=30)
    sig = s.evaluate(_make_overlay(rsi_vals), subplots)
    assert sig.direction == "偏多"


def test_rsi_overbought_bearish():
    rsi_vals = [68, 69, 70, 71, 72, 71]
    subplots = {"rsi": pd.DataFrame({"RSI": rsi_vals})}
    s = RSIThreshold(period=14, overbought=70, oversold=30)
    sig = s.evaluate(_make_overlay(rsi_vals), subplots)
    assert sig.direction == "偏空"


def test_rsi_cross_into_oversold():
    rsi_vals = [35, 34, 33, 32, 31, 29]
    subplots = {"rsi": pd.DataFrame({"RSI": rsi_vals})}
    s = RSIThreshold(period=14, overbought=70, oversold=30)
    sig = s.evaluate(_make_overlay(rsi_vals), subplots)
    assert sig.direction == "偏多"


def test_rsi_cross_into_overbought():
    rsi_vals = [65, 66, 67, 68, 69, 71]
    subplots = {"rsi": pd.DataFrame({"RSI": rsi_vals})}
    s = RSIThreshold(period=14, overbought=70, oversold=30)
    sig = s.evaluate(_make_overlay(rsi_vals), subplots)
    assert sig.direction == "偏空"


def test_macd_bullish_cross():
    macd_vals = [-1, -0.8, -0.6, -0.4, -0.6, 0.1]
    sig_vals = [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5]
    subplots = {"macd": pd.DataFrame({"MACD": macd_vals, "Signal": sig_vals, "Histogram": [0]*6})}
    s = MACDCross()
    sig = s.evaluate(_make_overlay([100]*6), subplots)
    assert sig.direction == "偏多"


def test_macd_bearish_cross():
    macd_vals = [0.5, 0.4, 0.3, 0.4, 0.3, -0.1]
    sig_vals = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    subplots = {"macd": pd.DataFrame({"MACD": macd_vals, "Signal": sig_vals, "Histogram": [0]*6})}
    s = MACDCross()
    sig = s.evaluate(_make_overlay([100]*6), subplots)
    assert sig.direction == "偏空"


def test_composite_strategy_needs_real_data():
    close = [100, 102, 104, 106, 108, 110]
    overlay = pd.DataFrame({
        "close": close, "SMA_20": [101]*6, "SMA_50": [102]*6, "SMA_200": [100]*6,
        "EMA_12": [103]*6, "EMA_26": [102]*6,
        "BB_upper": [110]*6, "BB_middle": [105]*6, "BB_lower": [100]*6,
    })
    subplots = {
        "rsi": pd.DataFrame({"RSI": [55, 56, 57, 58, 59, 60]}),
        "macd": pd.DataFrame({"MACD": [1]*6, "Signal": [0.5]*6, "Histogram": [0.5]*6}),
        "stoch": pd.DataFrame({"%K": [50]*6, "%D": [45]*6}),
        "obv": pd.DataFrame({"OBV": [1000]*6}),
    }
    s = CompositeStrategy()
    sig = s.evaluate(overlay, subplots)
    assert sig.direction in ("偏多", "偏空", "中立")


def test_custom_composite_ma_only():
    configs = {
        "ma_cross": {"enabled": True, "params": {"fast": 12, "slow": 26, "type": "ema"}, "weight": 1},
        "rsi": {"enabled": False, "params": {}, "weight": 1},
        "macd": {"enabled": False, "params": {}, "weight": 1},
        "composite": {"enabled": False, "params": {}, "weight": 1},
    }
    cc = CustomComposite(configs, threshold=0.6)
    close = [100, 101, 102, 103, 104, 105]
    fast = [103, 103, 103, 103, 102, 104]
    slow = [103, 103, 103, 103, 103, 103]
    overlay = pd.DataFrame({"close": close, "EMA_12": fast, "EMA_26": slow,
                            "SMA_20": close, "SMA_50": close, "SMA_200": close})
    sig = cc.evaluate(overlay, {})
    assert sig.direction == "偏多"
    assert sig.confidence > 0.6
