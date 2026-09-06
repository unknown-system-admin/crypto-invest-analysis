import numpy as np
import pandas as pd
from bot.signals import evaluate, momentum_series, htf_direction_from_score
from bot.signals import volatility_ok, latest_atr


def _ohlcv(n, trend=0.0001, base=40000.0):
    idx = pd.date_range("2026-01-01", periods=n, freq="5min")
    close = base * (1 + trend) ** np.arange(n)
    return pd.DataFrame({
        "open": close * 0.999, "high": close * 1.002, "low": close * 0.998,
        "close": close, "volume": 100.0,
    }, index=idx)


def test_momentum_series_returns_bounded_scores():
    df = _ohlcv(300)
    s = momentum_series(df)
    assert s.dropna().between(-1, 1).all()
    assert len(s) == len(df)


def test_htf_direction_thresholds():
    assert htf_direction_from_score(0.5, 0.1) == "long"
    assert htf_direction_from_score(-0.5, 0.1) == "short"
    assert htf_direction_from_score(0.05, 0.1) == "flat"


def test_evaluate_enter_long_on_up_cross_in_bull():
    df_htf = _ohlcv(300, trend=0.0005)          # rising 1h -> bullish HTF
    df_ltf = _ohlcv(300, trend=0.0005)          # rising 5m
    sig = evaluate(df_htf, df_ltf, 0.10, side="flat")
    assert sig.action in ("enter_long", "none")
    assert sig.htf_score >= -1.0


def test_evaluate_enter_short_in_bear():
    df_htf = _ohlcv(300, trend=-0.0005)         # falling -> bearish HTF
    df_ltf = _ohlcv(300, trend=-0.0005)
    sig = evaluate(df_htf, df_ltf, 0.10, side="flat")
    assert sig.action in ("enter_short", "none")


def test_evaluate_enter_long_fires_on_up_cross(monkeypatch):
    import bot.signals as sigs
    calls = {"n": 0}

    def fake_momentum(df):
        calls["n"] += 1
        if calls["n"] == 1:  # HTF call -> bullish
            return pd.Series([0.5], index=[0])
        return pd.Series([0.0, -0.1, 0.1], index=[0, 1, 2])  # LTF scores

    monkeypatch.setattr(sigs, "momentum_series", fake_momentum)
    sig = sigs.evaluate(None, None, 0.10, side="flat")
    assert sig.action == "enter_long"


def test_evaluate_enter_short_fires_on_down_cross(monkeypatch):
    import bot.signals as sigs
    calls = {"n": 0}

    def fake_momentum(df):
        calls["n"] += 1
        if calls["n"] == 1:  # HTF -> bearish
            return pd.Series([-0.5], index=[0])
        return pd.Series([0.0, 0.1, -0.1], index=[0, 1, 2])  # LTF: prev delta +0.1, curr -0.1

    monkeypatch.setattr(sigs, "momentum_series", fake_momentum)
    sig = sigs.evaluate(None, None, 0.10, side="flat")
    assert sig.action == "enter_short"


def test_evaluate_none_when_neutral_momentum(monkeypatch):
    import bot.signals as sigs

    def fake_momentum(df):
        return pd.Series([0.0, 0.0, 0.0], index=[0, 1, 2])  # zero deltas

    monkeypatch.setattr(sigs, "momentum_series", fake_momentum)
    sig = sigs.evaluate(None, None, 0.10, side="flat")
    assert sig.action == "none"


def test_evaluate_exit_long_when_direction_flips(monkeypatch):
    import bot.signals as sigs
    calls = {"n": 0}

    def fake_momentum(df):
        calls["n"] += 1
        if calls["n"] == 1:  # HTF now bearish while holding long
            return pd.Series([-0.5], index=[0])
        return pd.Series([0.1, 0.1, 0.1], index=[0, 1, 2])

    monkeypatch.setattr(sigs, "momentum_series", fake_momentum)
    sig = sigs.evaluate(None, None, 0.10, side="long")
    assert sig.action == "exit_long"


def test_latest_atr_returns_positive():
    df = _ohlcv(300, trend=0.0005)
    assert latest_atr(df) > 0


def test_volatility_ok_high_vol():
    df = _ohlcv(300, trend=0.0005)
    assert volatility_ok(df, min_atr_pct=0.0001) is True


def test_volatility_ok_low_vol_blocked():
    idx = pd.date_range("2026-01-01", periods=300, freq="5min")
    close = pd.Series([40000.0] * 300, index=idx)  # perfectly flat -> ATR ~0
    df = pd.DataFrame({"open": close, "high": close, "low": close,
                       "close": close, "volume": 100.0}, index=idx)
    assert volatility_ok(df, min_atr_pct=0.0001) is False