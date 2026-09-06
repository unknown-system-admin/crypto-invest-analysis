import numpy as np
import pandas as pd
from bot.meanreversion import latest_rsi, evaluate_meanrev


def _ohlcv(n, trend, base=40000.0):
    idx = pd.date_range("2026-01-01", periods=n, freq="5min")
    close = base * (1 + trend) ** np.arange(n)
    return pd.DataFrame({"open": close*0.999, "high": close*1.002, "low": close*0.998,
                         "close": close, "volume": 100.0}, index=idx)


def test_latest_rsi_bounded():
    df = _ohlcv(300, trend=0.0005)
    assert 0 <= latest_rsi(df).iloc[-1] <= 100


def test_evaluate_meanrev_entry_long_oversold_in_uptrend(monkeypatch):
    import bot.meanreversion as mr
    calls = {"n": 0}

    def fake_rsi(df):
        calls["n"] += 1
        return pd.Series([40.0, 40.0, 24.0], index=[0, 1, 2])  # prev 40 > 25, curr 24 <= 25

    monkeypatch.setattr(mr, "latest_rsi", fake_rsi)
    monkeypatch.setattr(mr, "htf_trend_up", lambda df_htf: True)
    monkeypatch.setattr(mr, "htf_trend_down", lambda df_htf: False)
    sig = evaluate_meanrev(None, None, oversold=25, overbought=75, side="flat")
    assert sig.action == "enter_long"


def test_evaluate_meanrev_blocks_neutral_rsi(monkeypatch):
    import bot.meanreversion as mr
    monkeypatch.setattr(mr, "latest_rsi", lambda df: pd.Series([50.0, 50.0]))
    monkeypatch.setattr(mr, "htf_trend_up", lambda df_htf: True)
    monkeypatch.setattr(mr, "htf_trend_down", lambda df_htf: False)
    assert evaluate_meanrev(None, None, 25, 75, "flat").action == "none"


def test_evaluate_meanrev_exit_long_on_reversion(monkeypatch):
    import bot.meanreversion as mr
    monkeypatch.setattr(mr, "latest_rsi", lambda df: pd.Series([50.0, 58.0]))  # >= exit 55
    sig = evaluate_meanrev(None, None, 25, 75, side="long")
    assert sig.action == "exit_long"


def test_evaluate_meanrev_entry_requires_funding_alignment(monkeypatch):
    import bot.meanreversion as mr
    # RSI oversold (would enter_long), but funding bias = "short" (crowded longs)
    monkeypatch.setattr(mr, "latest_rsi", lambda df: pd.Series([40.0, 40.0, 24.0]))
    monkeypatch.setattr(mr, "htf_trend_up", lambda df_htf: True)
    monkeypatch.setattr(mr, "htf_trend_down", lambda df_htf: False)
    sig = evaluate_meanrev(None, None, 25, 75, "flat", funding_bias="short")
    assert sig.action == "none"   # blocked: funding says short, RSI says buy -> conflict


def test_evaluate_meanrev_entry_allows_funding_aligned(monkeypatch):
    import bot.meanreversion as mr
    monkeypatch.setattr(mr, "latest_rsi", lambda df: pd.Series([40.0, 40.0, 24.0]))
    monkeypatch.setattr(mr, "htf_trend_up", lambda df_htf: True)
    monkeypatch.setattr(mr, "htf_trend_down", lambda df_htf: False)
    sig = evaluate_meanrev(None, None, 25, 75, "flat", funding_bias="long")
    assert sig.action == "enter_long"