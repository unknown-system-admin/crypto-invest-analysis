import pandas as pd
import numpy as np
from backtest_engine.rule_strategy import MomentumRuleStrategy


def test_momentum_rule_buy_signal():
    features = pd.Series({
        "momentum_score": 0.7,
        "momentum_delta": 0.1,
    })
    
    strategy = MomentumRuleStrategy(buy_threshold=0.6, sell_threshold=-0.6)
    signal = strategy.evaluate(features)
    
    assert signal.direction == "偏多"


def test_momentum_rule_sell_signal():
    features = pd.Series({
        "momentum_score": -0.7,
        "momentum_delta": -0.1,
    })
    
    strategy = MomentumRuleStrategy(buy_threshold=0.6, sell_threshold=-0.6)
    signal = strategy.evaluate(features)
    
    assert signal.direction == "偏空"


def test_momentum_rule_neutral():
    features = pd.Series({
        "momentum_score": 0.0,
        "momentum_delta": 0.0,
    })
    
    strategy = MomentumRuleStrategy(buy_threshold=0.6, sell_threshold=-0.6)
    signal = strategy.evaluate(features)
    
    assert signal.direction == "中立"


def test_long_holds_shallow_pullback_between_thresholds():
    # side="long": score -0.20 < short_entry (-0.15) but > long_exit (-0.30)
    # -> must HOLD (中立), NOT exit. Split thresholds in action.
    features = pd.Series({"momentum_score": -0.20, "momentum_delta": -0.1})
    strategy = MomentumRuleStrategy(
        buy_threshold=0.05, sell_threshold=-0.30, short_entry_threshold=-0.15)
    assert strategy.evaluate(features, side="long").direction == "中立"


def test_long_exit_fires_at_deep_threshold():
    features = pd.Series({"momentum_score": -0.40, "momentum_delta": -0.1})
    strategy = MomentumRuleStrategy(
        buy_threshold=0.05, sell_threshold=-0.30, short_entry_threshold=-0.15)
    assert strategy.evaluate(features, side="long").direction == "偏空"


def test_short_entry_fires_at_shallow_threshold_when_flat():
    # flat + score -0.20: opens short (below -0.15) even though above -0.30
    features = pd.Series({"momentum_score": -0.20, "momentum_delta": -0.1})
    strategy = MomentumRuleStrategy(
        buy_threshold=0.05, sell_threshold=-0.30, short_entry_threshold=-0.15)
    assert strategy.evaluate(features, side="flat").direction == "偏空"


def test_short_entry_blocked_above_shallow_threshold():
    features = pd.Series({"momentum_score": -0.10, "momentum_delta": -0.1})
    strategy = MomentumRuleStrategy(
        buy_threshold=0.05, sell_threshold=-0.30, short_entry_threshold=-0.15)
    assert strategy.evaluate(features, side="flat").direction == "中立"


def test_short_covers_on_bounce():
    features = pd.Series({"momentum_score": 0.2, "momentum_delta": 0.1})
    strategy = MomentumRuleStrategy(
        buy_threshold=0.05, sell_threshold=-0.30, short_entry_threshold=-0.15)
    assert strategy.evaluate(features, side="short").direction == "偏多"
