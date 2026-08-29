import numpy as np
from backtest_engine.metrics import calculate_metrics, format_report


def test_calculate_metrics():
    equity_curve = [10000, 10100, 10200, 9900, 10300, 10500]
    
    metrics = calculate_metrics(equity_curve, initial_capital=10000)
    
    assert "total_return" in metrics
    assert "max_drawdown" in metrics
    assert "sharpe_ratio" in metrics
    assert "win_rate" in metrics
    assert "profit_factor" in metrics
    assert "calmar_ratio" in metrics
    assert "sortino_ratio" in metrics


def test_calculate_metrics_with_trades():
    equity_curve = [10000, 10100, 10200, 9900, 10300, 10500]
    trades = [
        {"action": "buy", "price": 100},
        {"action": "sell", "price": 105, "pnl": 50},
        {"action": "buy", "price": 103},
        {"action": "sell", "price": 99, "pnl": -40},
    ]
    
    metrics = calculate_metrics(equity_curve, initial_capital=10000, trades=trades)
    
    assert metrics["total_trades"] == 2
    assert metrics["win_rate"] > 0


def test_format_report():
    metrics = {
        "total_return": 5.0,
        "max_drawdown": 3.0,
        "sharpe_ratio": 1.2,
        "win_rate": 60.0,
        "profit_factor": 1.5,
        "calmar_ratio": 1.7,
        "sortino_ratio": 1.8,
        "total_trades": 2,
    }
    
    report = format_report(metrics)
    
    assert isinstance(report, str)
    assert "5.0%" in report
    assert "3.0%" in report
    assert "1.2" in report
    assert "60.0%" in report
