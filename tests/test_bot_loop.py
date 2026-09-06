import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from bot.config import BotConfig
from bot.loop import _equity_estimate, run_bot


def _ohlcv(n, trend):
    idx = pd.date_range("2026-01-01", periods=n, freq="5min")
    close = 40000.0 * (1 + trend) ** np.arange(n)
    return pd.DataFrame({"open": close * 0.999, "high": close * 1.002,
                         "low": close * 0.998, "close": close, "volume": 100.0}, index=idx)


def _flat(n, base):
    idx = pd.date_range("2026-01-01", periods=n, freq="5min")
    close = np.ones(n) * base
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


def test_new_day_resets_daily_loss_stop(tmp_path):
    """A session left stopped for daily loss must not stay stopped when the
    bot is relaunched on a new UTC day: session fields reset, positions kept."""
    logs = []
    cfg = BotConfig(dry_run=True, poll_seconds=0, symbols=["BTC/USDT:USDT"])
    state_path = tmp_path / "state.json"
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    state = {"session_start_equity": 10000.0, "started_at": yesterday,
             "daily_loss_stopped": True,
             "positions": {"BTC/USDT:USDT": None}, "realized_pnl": -1500.0}
    state_path.write_text(json.dumps(state))

    fetcher = FakeFetcher(htf=_ohlcv(cfg.htf_candles, 0.0),
                          ltf=_ohlcv(cfg.ltf_candles, 0.0))
    run_bot(cfg, executor=None, fetch_fn=fetcher, state_path=state_path,
            max_iterations=3, logger=logs.append)

    new_state = __import__("bot.state", fromlist=["load_state"]).load_state(state_path)
    assert new_state["daily_loss_stopped"] is False
    assert new_state["realized_pnl"] == 0.0
    started = datetime.fromisoformat(new_state["started_at"])
    assert started.date() == datetime.now(timezone.utc).date()
    assert any("New session day detected" in line for line in logs)


def test_new_day_reset_baselines_to_cash_equity(tmp_path):
    """New-day reset re-baselines session_start_equity to cash equity (start +
    realized only, excluding unrealized), preserving each open position's
    absolute unrealized mark so equity estimates stay drift-free: start +
    realized + sum(abs marks) = true equity on every tick."""
    logs = []
    cfg = BotConfig(dry_run=True, poll_seconds=0, symbols=["BTC/USDT:USDT"])
    state_path = tmp_path / "state.json"
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    state = {"session_start_equity": 10000.0, "started_at": yesterday,
             "daily_loss_stopped": True, "realized_pnl": -500.0,
             "positions": {"BTC/USDT:USDT": {"side": "long", "entry_price": 40000.0,
                                             "qty": 0.05, "entry_time": "2026-09-05T10:00:00",
                                             "unrealized": 50.0}}}
    state_path.write_text(json.dumps(state))

    fetcher = FakeFetcher(htf=_flat(cfg.htf_candles, 41000.0),
                          ltf=_flat(cfg.ltf_candles, 41000.0))
    run_bot(cfg, executor=None, fetch_fn=fetcher, state_path=state_path,
            max_iterations=0, logger=logs.append)

    new_state = __import__("bot.state", fromlist=["load_state"]).load_state(state_path)
    assert new_state["session_start_equity"] == 10000.0 - 500.0
    assert new_state["realized_pnl"] == 0.0
    assert new_state["daily_loss_stopped"] is False
    pos = new_state["positions"]["BTC/USDT:USDT"]
    assert pos is not None
    assert pos["unrealized"] == 50.0
    assert _equity_estimate(new_state) == 10000.0 - 500.0 + 50.0


def test_equity_sums_across_positions():
    """Portfolio equity = session start + realized + sum of each open
    position's stored unrealized; positions without a stored value count 0."""
    state = {
        "session_start_equity": 10000.0,
        "realized_pnl": 200.0,
        "positions": {
            "BTC/USDT:USDT": {"side": "long", "entry_price": 40000.0, "qty": 0.05,
                              "unrealized": -25.0},
            "SOL/USDT:USDT": {"side": "short", "entry_price": 100.0, "qty": 5.0,
                              "unrealized": 15.0},
            "ETH/USDT:USDT": {"side": "long", "entry_price": 50.0, "qty": 2.0},
        },
    }
    assert _equity_estimate(state) == 10190.0