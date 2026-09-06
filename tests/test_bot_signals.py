import numpy as np
import pandas as pd
from bot.signals import evaluate, momentum_series, htf_direction_from_score


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


def test_evaluate_exit_long_when_direction_flips():
    # side="long" but HTF turns bearish -> exit_long
    df_htf = _ohlcv(300, trend=-0.0005)
    df_ltf = _ohlcv(300, trend=0.0005)
    sig = evaluate(df_htf, df_ltf, 0.10, side="long")
    assert sig.action == "exit_long"


def test_evaluate_none_in_neutral_market():
    df_htf = _ohlcv(300, trend=0.0)
    df_ltf = _ohlcv(300, trend=0.0)
    sig = evaluate(df_htf, df_ltf, 0.10, side="flat")
    assert sig.action == "none"