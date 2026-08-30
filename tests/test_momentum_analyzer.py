import pandas as pd
import numpy as np
from analyzers.momentum_analyzer import MomentumAnalyzer


def test_momentum_analyzer_correlation():
    dates = pd.date_range("2024-01-01", periods=300, freq="1D")
    np.random.seed(42)
    close = 50000 + np.cumsum(np.random.randn(300) * 500)
    
    df = pd.DataFrame({
        "close": close,
        "momentum_score": np.sin(np.linspace(0, 10, 300)),
    }, index=dates)
    
    analyzer = MomentumAnalyzer()
    result = analyzer.analyze_correlation(df, forward_days=5)
    
    assert "correlation" in result
    assert "p_value" in result
    assert isinstance(result["correlation"], float)


def test_momentum_analyzer_threshold_backtest():
    dates = pd.date_range("2024-01-01", periods=300, freq="1D")
    np.random.seed(42)
    close = 50000 + np.cumsum(np.random.randn(300) * 500)
    
    df = pd.DataFrame({
        "close": close,
        "momentum_score": np.sin(np.linspace(0, 10, 300)),
        "momentum_delta": np.cos(np.linspace(0, 10, 300)),
    }, index=dates)
    
    analyzer = MomentumAnalyzer()
    results = analyzer.backtest_thresholds(
        df,
        buy_thresholds=[0.2, 0.3, 0.4],
        sell_thresholds=[-0.2, -0.3, -0.4],
    )
    
    assert len(results) == 9  # 3x3 combinations
    assert all("buy_threshold" in r for r in results)
    assert all("total_return_pct" in r for r in results)
