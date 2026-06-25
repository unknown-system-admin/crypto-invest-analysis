import pandas as pd
from trading.strategy import RSITrendFilter, Signal


def _make_overlay(close_prices, rsi_values):
    n = len(close_prices)
    dates = pd.date_range("2025-01-01", periods=n, freq="4h")
    overlay = pd.DataFrame({
        "open": [100.0] * n,
        "high": [max(100.0, c) for c in close_prices],
        "low": [min(100.0, c) for c in close_prices],
        "close": close_prices,
        "volume": [1000.0] * n,
    }, index=dates)
    subplots = {"rsi": pd.DataFrame({"RSI": rsi_values}, index=dates)}
    return overlay, subplots


def test_rsi_trend_bullish_only_in_uptrend():
    close = [90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101]
    rsi = [50, 48, 45, 42, 38, 35, 33, 30, 32, 33, 35, 28]
    overlay, subplots = _make_overlay(close, rsi)
    strat = RSITrendFilter(period=14, overbought=70, oversold=30, trend_period=5, trend_type="sma")
    sig = strat.evaluate(overlay, subplots)
    assert sig.direction == "偏多", f"Expected 偏多, got {sig.direction}"
    assert sig.confidence > 0


def test_rsi_trend_bearish_only_in_downtrend():
    close = [110, 108, 106, 104, 102, 100, 98, 96, 94, 92, 90, 88]
    rsi = [55, 58, 62, 65, 68, 72, 70, 68, 65, 62, 68, 75]
    overlay, subplots = _make_overlay(close, rsi)
    strat = RSITrendFilter(period=14, overbought=70, oversold=30, trend_period=5, trend_type="sma")
    sig = strat.evaluate(overlay, subplots)
    assert sig.direction == "偏空", f"Expected 偏空, got {sig.direction}"
    assert sig.confidence > 0


def test_rsi_trend_blocks_bullish_in_downtrend():
    close = [105, 104, 103, 102, 101, 100, 99, 98, 97, 96, 95, 94]
    rsi = [50, 48, 45, 42, 38, 35, 33, 30, 32, 33, 35, 28]
    overlay, subplots = _make_overlay(close, rsi)
    strat = RSITrendFilter(period=14, overbought=70, oversold=30, trend_period=5, trend_type="sma")
    sig = strat.evaluate(overlay, subplots)
    assert sig.direction == "中立", f"Expected 中立, got {sig.direction}"


def test_rsi_trend_blocks_bearish_in_uptrend():
    close = [90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112]
    rsi = [55, 58, 62, 65, 68, 72, 70, 68, 65, 62, 68, 75]
    overlay, subplots = _make_overlay(close, rsi)
    strat = RSITrendFilter(period=14, overbought=70, oversold=30, trend_period=5, trend_type="sma")
    sig = strat.evaluate(overlay, subplots)
    assert sig.direction == "中立", f"Expected 中立, got {sig.direction}"


def test_rsi_trend_uses_ema_when_specified():
    close = [90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101]
    rsi = [50, 48, 45, 42, 38, 35, 33, 30, 32, 33, 35, 28]
    overlay, subplots = _make_overlay(close, rsi)
    strat = RSITrendFilter(period=14, overbought=70, oversold=30, trend_period=5, trend_type="ema")
    sig = strat.evaluate(overlay, subplots)
    assert sig.direction == "偏多", f"Expected 偏多, got {sig.direction}"
