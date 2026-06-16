from trading.backtest import _calc_sharpe, _calc_max_drawdown
import pandas as pd

def test_sharpe_positive():
    rets = pd.Series([0.001] * 100)
    sr = _calc_sharpe(rets, risk_free=0.02)
    assert sr > 0

def test_sharpe_negative():
    rets = pd.Series([-0.001] * 100)
    sr = _calc_sharpe(rets, risk_free=0.02)
    assert sr < 0

def test_sharpe_insufficient_data():
    rets = pd.Series([0.01])
    sr = _calc_sharpe(rets)
    assert sr == 0.0

def test_max_drawdown():
    eq = [100, 110, 90, 80, 95, 105]
    dd = _calc_max_drawdown(eq)
    assert dd > 0
    assert dd < 100

def test_max_drawdown_zero():
    eq = [100, 100, 100]
    dd = _calc_max_drawdown(eq)
    assert dd == 0.0

def test_max_drawdown_empty():
    dd = _calc_max_drawdown([])
    assert dd == 0.0

def test_run_backtest_with_defaults():
    from trading.backtest import run_backtest
    try:
        result = run_backtest(symbol="BTC/USDT", timeframe="1h", limit=200)
        assert result.symbol == "BTC/USDT"
        assert result.initial_capital == 10000
        assert result.total_trades >= 0
        assert len(result.equity_curve) > 0
        assert result.final_equity > 0
    except (ValueError, ConnectionError) as e:
        pass
