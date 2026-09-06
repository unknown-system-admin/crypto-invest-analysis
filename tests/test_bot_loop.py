import numpy as np
import pandas as pd
from bot.config import BotConfig
from bot.loop import run_bot


def _ohlcv(n, trend):
    idx = pd.date_range("2026-01-01", periods=n, freq="5min")
    close = 40000.0 * (1 + trend) ** np.arange(n)
    return pd.DataFrame({"open": close * 0.999, "high": close * 1.002,
                         "low": close * 0.998, "close": close, "volume": 100.0}, index=idx)


def _v_ohlcv(n, dip=0.01, up=0.01):
    """Flat series ending in a dip then recovery, forcing an up-cross on the
    last LTF delta so enter_long fires deterministically (constant-trend
    series saturate momentum, leaving delta == 0 at the window end)."""
    idx = pd.date_range("2026-01-01", periods=n, freq="5min")
    close = np.ones(n) * 40000.0
    close[-2] = 40000.0 * (1 - dip)
    close[-1] = 40000.0 * (1 + up)
    return pd.DataFrame({"open": close * 0.999, "high": close * 1.002,
                         "low": close * 0.998, "close": close, "volume": 100.0}, index=idx)


class FakeFetcher:
    def __init__(self, htf, ltf):
        self.htf, self.ltf = htf, ltf

    def __call__(self, symbol, timeframe, limit):
        return self.htf if timeframe == "1h" else self.ltf


def test_dry_run_opens_virtual_long(tmp_path):
    logs = []
    cfg = BotConfig(dry_run=True, poll_seconds=0, symbols=["BTC/USDT:USDT"])
    fetcher = FakeFetcher(htf=_ohlcv(cfg.htf_candles, 0.0005),
                          ltf=_v_ohlcv(cfg.ltf_candles))
    run_bot(cfg, executor=None, fetch_fn=fetcher, state_path=tmp_path / "state.json",
            max_iterations=20, logger=logs.append)
    state = __import__("bot.state", fromlist=["load_state"]).load_state(tmp_path / "state.json")
    pos = state["positions"]["BTC/USDT:USDT"]
    assert pos is not None
    assert pos["side"] == "long"
    assert any("enter_long" in line for line in logs)


def test_dry_run_no_trades_in_neutral_market(tmp_path):
    logs = []
    cfg = BotConfig(dry_run=True, poll_seconds=0, symbols=["BTC/USDT:USDT"])
    fetcher = FakeFetcher(htf=_ohlcv(cfg.htf_candles, 0.0),
                          ltf=_ohlcv(cfg.ltf_candles, 0.0))
    run_bot(cfg, executor=None, fetch_fn=fetcher, state_path=tmp_path / "state.json",
            max_iterations=10, logger=logs.append)
    state = __import__("bot.state", fromlist=["load_state"]).load_state(tmp_path / "state.json")
    assert state["positions"]["BTC/USDT:USDT"] is None


def test_dry_run_exits_long_when_direction_flips(tmp_path):
    logs = []
    cfg = BotConfig(dry_run=True, poll_seconds=0, symbols=["BTC/USDT:USDT"])
    calls = {"n": 0}

    def flipper(symbol, timeframe, limit):
        if timeframe == "1h":
            calls["n"] += 1
            trend = 0.0005 if calls["n"] <= 5 else -0.0005
            return _ohlcv(limit, trend)
        return _v_ohlcv(limit)

    run_bot(cfg, executor=None, fetch_fn=flipper, state_path=tmp_path / "state.json",
            max_iterations=15, logger=logs.append)
    state = __import__("bot.state", fromlist=["load_state"]).load_state(tmp_path / "state.json")
    pos = state["positions"]["BTC/USDT:USDT"]
    assert pos is None  # exited
    assert any("exit_long" in line for line in logs)