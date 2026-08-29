import pandas as pd
import numpy as np
from backtest_engine.engine import BacktestEngine
from backtest_engine.strategy import Strategy, Signal


class DummyStrategy(Strategy):
    def evaluate(self, features: pd.Series) -> Signal:
        if features.get("momentum_score", 0) > 0.5:
            return Signal("偏多", 0.8, "dummy")
        elif features.get("momentum_score", 0) < -0.5:
            return Signal("偏空", 0.8, "dummy")
        return Signal("中立", 0.5, "dummy")


def test_backtest_engine_runs():
    dates = pd.date_range("2024-01-01", periods=200, freq="1h")
    features = pd.DataFrame({
        "momentum_score": np.sin(np.linspace(0, 10, 200)),
        "close": 40000 + np.sin(np.linspace(0, 10, 200)) * 1000,
    }, index=dates)
    
    engine = BacktestEngine(
        strategy=DummyStrategy(),
        initial_capital=10000,
    )
    
    result = engine.run(features)
    
    assert result.total_trades >= 0
    assert result.final_equity > 0


def test_backtest_engine_tracks_positions():
    dates = pd.date_range("2024-01-01", periods=100, freq="1h")
    features = pd.DataFrame({
        "momentum_score": [0.8] * 50 + [-0.8] * 50,
        "close": [40000] * 50 + [39000] * 50,
    }, index=dates)
    
    engine = BacktestEngine(
        strategy=DummyStrategy(),
        initial_capital=10000,
    )
    
    result = engine.run(features)
    
    assert result.total_trades >= 1
