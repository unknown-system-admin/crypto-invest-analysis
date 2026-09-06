import pandas as pd
import numpy as np
from monitor.signal_bot import compute_latest_signal, load_signal_state, save_signal_state


def _make_features(dates, closes, scores, deltas, sma200, sma50):
    return pd.DataFrame({
        "momentum_score": scores,
        "momentum_delta": deltas,
        "close": closes,
        "SMA_200": sma200,
        "SMA_50": sma50,
    }, index=pd.to_datetime(dates))


def test_latest_trade_returns_newest_entry():
    dates = pd.date_range("2026-01-01", periods=10, freq="1D").strftime("%Y-%m-%d")
    # flat -> long at bar 5 (score>0.05, delta>0, close>SMA_200), hold
    scores = [-0.2]*5 + [0.3]*5
    deltas = [-0.05]*5 + [0.05]*5
    closes = [40000.0]*10
    features = _make_features(dates, closes, scores, deltas, sma200=[39000.0]*10, sma50=[39500.0]*10)
    sig = compute_latest_signal(features, buy=0.05, sell=-0.30, trend_filter=True, cooldown=3)
    assert sig is not None
    assert sig["action"] == "buy"
    assert sig["date"] == dates[5]


def test_latest_trade_detects_short():
    dates = pd.date_range("2026-01-01", periods=10, freq="1D").strftime("%Y-%m-%d")
    scores = [0.2]*5 + [-0.4]*5
    deltas = [0.05]*5 + [-0.05]*5
    closes = [40000.0]*10
    features = _make_features(dates, closes, scores, deltas, sma200=[41000.0]*10, sma50=[40500.0]*10)
    sig = compute_latest_signal(features, buy=0.05, sell=-0.30, trend_filter=True, cooldown=3)
    assert sig is not None
    assert sig["action"] == "short_sell"


def test_no_signal_when_no_trades():
    dates = pd.date_range("2026-01-01", periods=10, freq="1D").strftime("%Y-%m-%d")
    features = _make_features(dates, [40000.0]*10, [0.0]*10, [0.0]*10,
                              sma200=[40000.0]*10, sma50=[40000.0]*10)
    sig = compute_latest_signal(features, buy=0.05, sell=-0.30, trend_filter=True, cooldown=3)
    assert sig is None


def test_signal_state_roundtrip(tmp_path):
    p = tmp_path / "sig.json"
    save_signal_state({"BTC/USDT": {"date": "2026-09-05", "action": "buy"}}, path=p)
    s = load_signal_state(path=p)
    assert s["BTC/USDT"]["date"] == "2026-09-05"
    assert s["BTC/USDT"]["action"] == "buy"


def test_trades_carry_date_and_score():
    """Engine regression: every trade dict has date + triggering score."""
    from backtest_engine.engine import BacktestEngine
    from backtest_engine.rule_strategy import MomentumRuleStrategy

    dates = pd.date_range("2026-01-01", periods=12, freq="1D")
    features = pd.DataFrame({
        "momentum_score": [-0.2, -0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
        "momentum_delta": [-0.05, -0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
        "close": [40000.0]*12,
        "SMA_200": [39000.0]*12,
        "SMA_50": [39500.0]*12,
    }, index=dates)
    engine = BacktestEngine(
        strategy=MomentumRuleStrategy(buy_threshold=0.05, sell_threshold=-0.30),
        max_position_pct=50, trend_filter=True, cooldown_bars=3,
    )
    r = engine.run(features)
    assert len(r.trades) >= 1
    t = r.trades[-1]
    assert "date" in t and "score" in t
    assert t["date"] == "2026-01-03"