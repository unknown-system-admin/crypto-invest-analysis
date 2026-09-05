import pandas as pd
import numpy as np
from backtest_engine.engine import BacktestEngine
from backtest_engine.strategy import Strategy, Signal


class AlwaysLong(Strategy):
    def evaluate(self, features: pd.Series) -> Signal:
        return Signal("偏多", 0.8, "test")


class AlwaysShort(Strategy):
    def evaluate(self, features: pd.Series) -> Signal:
        return Signal("偏空", 0.8, "test")


class FlipAfter(Strategy):
    """Long signal for first `n` bars, then short forever."""
    def __init__(self, n: int):
        self.n = n
        self.count = 0

    def evaluate(self, features: pd.Series) -> Signal:
        self.count += 1
        if self.count <= self.n:
            return Signal("偏多", 0.8, "test")
        return Signal("偏空", 0.8, "test")


class Alternating(Strategy):
    def __init__(self):
        self.count = 0

    def evaluate(self, features: pd.Series) -> Signal:
        self.count += 1
        if self.count % 2 == 1:
            return Signal("偏多", 0.8, "test")
        return Signal("偏空", 0.8, "test")


def _make_features(closes, sma200=None):
    dates = pd.date_range("2024-01-01", periods=len(closes), freq="1h")
    data = {"momentum_score": [0.9] * len(closes), "close": closes}
    if sma200 is not None:
        data["SMA_200"] = sma200
    return pd.DataFrame(data, index=dates)


def test_trend_filter_blocks_long_below_sma200():
    features = _make_features([40000.0] * 50, sma200=[50000.0] * 50)
    engine = BacktestEngine(
        strategy=AlwaysLong(),
        trend_filter=True,
        max_position_pct=50,
    )
    result = engine.run(features)
    assert result.total_trades == 0


def test_trend_filter_allows_long_above_sma200():
    features = _make_features([50000.0] * 50, sma200=[40000.0] * 50)
    engine = BacktestEngine(
        strategy=AlwaysLong(),
        trend_filter=True,
        max_position_pct=50,
    )
    result = engine.run(features)
    assert result.total_trades >= 1


def test_trend_filter_blocks_short_above_sma200():
    features = _make_features([50000.0] * 50, sma200=[40000.0] * 50)
    engine = BacktestEngine(
        strategy=AlwaysShort(),
        trend_filter=True,
        max_position_pct=50,
    )
    result = engine.run(features)
    assert result.total_trades == 0


def test_trend_filter_allows_short_below_sma200():
    features = _make_features([40000.0] * 50, sma200=[50000.0] * 50)
    engine = BacktestEngine(
        strategy=AlwaysShort(),
        trend_filter=True,
        max_position_pct=50,
    )
    result = engine.run(features)
    assert result.total_trades >= 1


def test_trend_filter_ignores_missing_sma200():
    features = _make_features([40000.0] * 50)
    engine = BacktestEngine(
        strategy=AlwaysLong(),
        trend_filter=True,
        max_position_pct=100,
    )
    result = engine.run(features)
    # Missing SMA_200 -> filter cannot verify, block entries (conservative)
    assert result.total_trades == 0


def test_min_holding_blocks_early_exit():
    # Long for 2 bars, then short signal. min_holding=20 > total bars -> exit never fires
    closes = [40000.0, 40000.0, 39000.0, 38000.0, 37000.0, 36000.0, 35000.0, 34000.0, 33000.0, 32000.0]
    features = _make_features(closes)
    engine = BacktestEngine(
        strategy=FlipAfter(2),
        min_holding_bars=20,
        max_position_pct=50,
    )
    result = engine.run(features)
    # Position should still be open (no sell trade)
    sells = [t for t in result.trades if t["action"] == "sell" and t.get("reason") != "drawdown_stop"]
    assert len(sells) == 0


def test_min_holding_allows_exit_after_threshold():
    # Long for 10 bars, then short. min_holding=5 -> exit allowed at bar 5+
    closes = [40000.0] * 10 + [39000.0] * 10
    features = _make_features(closes)
    engine = BacktestEngine(
        strategy=FlipAfter(10),
        min_holding_bars=5,
        max_position_pct=50,
    )
    result = engine.run(features)
    sells = [t for t in result.trades if t["action"] == "sell"]
    assert len(sells) >= 1


def test_cooldown_blocks_early_reentry():
    # Alternating signals every bar. Without cooldown -> ~10 trades.
    # With cooldown=5 -> far fewer entries.
    closes = [40000.0] * 20
    features = _make_features(closes)

    engine_free = BacktestEngine(
        strategy=Alternating(),
        max_position_pct=50,
    )
    result_free = engine_free.run(features)

    engine_cd = BacktestEngine(
        strategy=Alternating(),
        cooldown_bars=5,
        max_position_pct=50,
    )
    result_cd = engine_cd.run(features)

    assert result_cd.total_trades < result_free.total_trades


def test_drawdown_stop_ignores_min_holding():
    # Enter long at 40000 with 95% capital, price crashes to 24000 (40%+ dd)
    # -> stop must fire even though min_holding is huge
    closes = [40000.0, 24000.0, 24000.0]
    features = _make_features(closes)
    engine = BacktestEngine(
        strategy=AlwaysLong(),
        min_holding_bars=100,
        max_drawdown_stop=30.0,
        max_position_pct=95,
    )
    result = engine.run(features)
    stops = [t for t in result.trades if t["action"] == "sell" and t.get("reason") == "drawdown_stop"]
    assert len(stops) >= 1


def test_default_filter_values_preserve_old_behavior():
    # Defaults: trend_filter=False, min_holding_bars=0, cooldown_bars=0
    # AlwaysLong with flat price -> enters once, never exits
    closes = [40000.0] * 30
    features = _make_features(closes)
    engine = BacktestEngine(
        strategy=AlwaysLong(),
        max_position_pct=50,
    )
    result = engine.run(features)
    assert result.total_trades == 1


def test_equity_correct_while_position_open():
    """Regression: equity must include position market value, not just PnL.

    Before the fix, equity = cash + unrealized_pnl understated equity by the
    full entry cost while a position was open, causing phantom drawdowns and
    false drawdown-stop triggers (the root cause of the -99.3% death spiral).
    """
    closes = [40000.0] * 5
    features = _make_features(closes)
    engine = BacktestEngine(
        strategy=AlwaysLong(),
        max_position_pct=50,
    )
    result = engine.run(features)
    # Entry bar equity: capital minus fees only (~0.15%), NOT minus position cost
    assert result.equity_curve[0] > 9900
    # Final equity: flat price -> only fees/slippage drag
    assert result.equity_curve[-1] > 9900
