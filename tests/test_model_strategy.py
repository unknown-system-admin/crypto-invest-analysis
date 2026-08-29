import pandas as pd
import numpy as np
from unittest.mock import Mock
from backtest_engine.model_strategy import ModelStrategy


def test_model_strategy_buy_signal():
    mock_model = Mock()
    mock_model.predict.return_value = np.array([1])
    
    features = pd.Series({"momentum_score": 0.5, "RSI": 60})
    
    strategy = ModelStrategy(model=mock_model, feature_columns=["momentum_score", "RSI"])
    signal = strategy.evaluate(features)
    
    assert signal.direction == "偏多"


def test_model_strategy_sell_signal():
    mock_model = Mock()
    mock_model.predict.return_value = np.array([0])
    
    features = pd.Series({"momentum_score": -0.5, "RSI": 40})
    
    strategy = ModelStrategy(model=mock_model, feature_columns=["momentum_score", "RSI"])
    signal = strategy.evaluate(features)
    
    assert signal.direction == "偏空"


def test_model_strategy_neutral_when_no_confidence():
    mock_model = Mock()
    mock_model.predict.return_value = np.array([1])
    mock_model.predict_proba.return_value = np.array([[0.51, 0.49]])
    
    features = pd.Series({"momentum_score": 0.0, "RSI": 50})
    
    strategy = ModelStrategy(
        model=mock_model,
        feature_columns=["momentum_score", "RSI"],
        confidence_threshold=0.6,
    )
    signal = strategy.evaluate(features)
    
    assert signal.direction == "中立"