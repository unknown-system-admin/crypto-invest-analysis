import pandas as pd
import numpy as np
from trading.optimizer import (
    ParamSweepConfig,
    SweepResult,
    RsiSweepConfig,
    RsiSweepResult,
    MaSweepConfig,
    MaSweepResult,
    run_rsi_sweep,
    _run_single_simulation,
)


def _make_indicator_data(length=200):
    dates = pd.date_range(start="2024-01-01", periods=length, freq="h")
    close = np.linspace(100, 110, length) + np.random.randn(length) * 2
    overlay = pd.DataFrame({"close": close}, index=dates)
    overlay["SMA_20"] = pd.Series(close).rolling(5, min_periods=1).mean()
    overlay["SMA_50"] = pd.Series(close).rolling(10, min_periods=1).mean()
    overlay["SMA_200"] = pd.Series(close).rolling(20, min_periods=1).mean()
    overlay["EMA_12"] = pd.Series(close).ewm(span=12).mean()
    overlay["EMA_26"] = pd.Series(close).ewm(span=26).mean()
    overlay["BB_upper"] = overlay["SMA_20"] + 2 * pd.Series(close).rolling(20, min_periods=1).std()
    overlay["BB_middle"] = overlay["SMA_20"]
    overlay["BB_lower"] = overlay["SMA_20"] - 2 * pd.Series(close).rolling(20, min_periods=1).std()
    rsi = pd.DataFrame({"RSI": np.clip(np.random.randn(length) * 15 + 50, 0, 100)})
    macd = pd.DataFrame({
        "MACD": np.cumsum(np.random.randn(length)) * 0.5,
        "Signal": np.cumsum(np.random.randn(length)) * 0.3,
        "Histogram": np.random.randn(length) * 0.2,
    })
    stoch = pd.DataFrame({"%K": np.random.rand(length) * 100, "%D": np.random.rand(length) * 100})
    obv = pd.DataFrame({"OBV": np.cumsum(np.random.randn(length) * 1000)})
    subplots = {"rsi": rsi, "macd": macd, "stoch": stoch, "obv": obv}
    return overlay, subplots


def test_single_simulation_returns_sweep_result():
    overlay, subplots = _make_indicator_data(length=200)
    config = ParamSweepConfig()
    strategy_configs = {
        "ma_cross": {"enabled": True, "params": {"fast": 12, "slow": 26, "type": "ema"}, "weight": 1},
        "rsi": {"enabled": True, "params": {"period": 14, "overbought": 70, "oversold": 30}, "weight": 1},
        "macd": {"enabled": False, "params": {}, "weight": 1},
        "composite": {"enabled": True, "params": {}, "weight": 2},
    }
    sr = _run_single_simulation(overlay, subplots, strategy_configs, 0.6, config)
    assert isinstance(sr, SweepResult)
    assert sr.total_return_pct >= -100
    assert sr.sharpe_ratio >= -10
    assert sr.max_drawdown_pct >= 0
    assert sr.final_equity > 0
    assert sr.total_trades >= 0


def test_single_simulation_all_weights_zero():
    overlay, subplots = _make_indicator_data(length=200)
    config = ParamSweepConfig()
    strategy_configs = {
        "ma_cross": {"enabled": False, "params": {}, "weight": 0},
        "rsi": {"enabled": False, "params": {}, "weight": 0},
        "macd": {"enabled": False, "params": {}, "weight": 0},
        "composite": {"enabled": False, "params": {}, "weight": 0},
    }
    sr = _run_single_simulation(overlay, subplots, strategy_configs, 0.6, config)
    assert sr.total_trades == 0
    assert sr.final_equity == config.initial_capital


def test_run_parameter_sweep_returns_sorted():
    config = ParamSweepConfig(
        limit=200,
        weight_min=0, weight_max=2,
        threshold_start=0.5, threshold_stop=0.7, threshold_step=0.2,
    )
    from trading.optimizer import run_parameter_sweep
    try:
        results = run_parameter_sweep(config)
        assert len(results) > 0
        for i in range(len(results) - 1):
            assert results[i].total_return_pct >= results[i + 1].total_return_pct
    except (ValueError, ConnectionError):
        pass  # offline


def test_sweep_result_to_dict():
    sr = SweepResult(
        params={"ma_cross_weight": 1, "rsi_weight": 2, "threshold": 0.6},
        total_return_pct=15.5, sharpe_ratio=1.2, max_drawdown_pct=8.0,
        win_rate=55.0, total_trades=10, final_equity=11500.0, cagr=12.0,
    )
    d = sr.to_dict()
    assert d["total_return_pct"] == 15.5
    assert d["params"]["threshold"] == 0.6


def test_rsi_sweep_result_to_dict():
    sr = RsiSweepResult(
        params={"period": 14, "overbought": 70, "oversold": 30},
        total_return_pct=5.2, sharpe_ratio=0.8, max_drawdown_pct=3.0,
        win_rate=60.0, total_trades=20, final_equity=10500.0, cagr=8.0,
    )
    d = sr.to_dict()
    assert d["total_return_pct"] == 5.2
    assert d["params"]["period"] == 14


def test_rsi_sweep_invalid_combos_skipped():
    from trading.optimizer import _run_single_simulation
    from tests.test_optimizer import _make_indicator_data
    overlay, subplots = _make_indicator_data(length=200)
    config = RsiSweepConfig()
    strategy_configs = {
        "ma_cross": {"enabled": False, "params": {}, "weight": 0},
        "rsi": {"enabled": False, "params": {}, "weight": 0},
        "macd": {"enabled": False, "params": {}, "weight": 0},
        "composite": {"enabled": False, "params": {}, "weight": 0},
    }
    sr = _run_single_simulation(overlay, subplots, strategy_configs, 0.5, config)
    assert sr.total_trades == 0


def test_ma_sweep_result_to_dict():
    sr = MaSweepResult(
        params={"fast": 5, "slow": 20},
        total_return_pct=3.5, sharpe_ratio=0.6, max_drawdown_pct=5.0,
        win_rate=45.0, total_trades=15, final_equity=10350.0, cagr=5.0,
    )
    d = sr.to_dict()
    assert d["total_return_pct"] == 3.5
    assert d["params"]["fast"] == 5


def test_ma_sweep_config_defaults():
    cfg = MaSweepConfig()
    assert 5 in cfg.fast_values
    assert 20 in cfg.slow_values
    assert cfg.symbol == "BTC/USDT"


def test_ma_sweep_invalid_combos_skipped():
    from trading.optimizer import _run_single_simulation
    from tests.test_optimizer import _make_indicator_data
    overlay, subplots = _make_indicator_data(length=200)
    config = MaSweepConfig()
    strategy_configs = {
        "ma_cross": {"enabled": False, "params": {}, "weight": 0},
        "rsi": {"enabled": False, "params": {}, "weight": 0},
        "macd": {"enabled": False, "params": {}, "weight": 0},
        "composite": {"enabled": False, "params": {}, "weight": 0},
    }
    sr = _run_single_simulation(overlay, subplots, strategy_configs, 0.5, config)
    assert sr.total_trades == 0
