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
