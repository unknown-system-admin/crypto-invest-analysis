# HFT 訊號改善 v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve the HFT bot's intraday signal with a long Binance history data source, ATR trailing stop, and volatility filter; re-calibrate on 6+ months of data to find params with net-positive, statistically meaningful returns.

**Architecture:** Add `bot/binance_data.py` (long-history fetch, calibration only). Extend `bot/risk.py` with ATR trailing stop, `bot/signals.py` with volatility filter + `latest_atr`, `bot/state.py` position `peak` field, `bot/loop.py` wiring, `bot/config.py` new params. Rebuild `bot/backtest_calibration.py` to grid-scan the new parameter space on Binance data.

**Tech Stack:** Python 3.9, pandas, ccxt (binance public + okx), existing `bot/` modules from v1.

**Spec:** `docs/superpowers/specs/2026-09-06-hft-signal-v2-design.md`

---

## File Structure

```
bot/
├── binance_data.py          # NEW: long-history OHLCV fetch + CSV cache (calibration only)
├── config.py                # MODIFY: + atr_stop_mult, min_atr_pct
├── signals.py               # MODIFY: + latest_atr(), volatility_ok()
├── risk.py                  # MODIFY: + trailing_stop_price(), check_trailing_stop()
├── state.py                 # (no change; positions already store arbitrary keys)
├── loop.py                  # MODIFY: trailing stop + vol filter + peak tracking
└── backtest_calibration.py  # MODIFY: Binance data + new param grid
```

Existing `bot/` v1 modules are on master. All work happens on a fresh worktree branch.

---

### Task 1: bot/binance_data.py (long-history fetcher)

**Files:**
- Create: `bot/binance_data.py`
- Test: `tests/test_binance_data.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_binance_data.py
import pandas as pd
from bot.binance_data import fetch_ohlcv_long, cache_path


class FakeBinance:
    def __init__(self):
        self._rows = []
        for i in range(1200):
            self._rows.append([1700000000000 + i * 900000,
                               100.0, 101.0, 99.0, 100.5, 1000.0])

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None):
        if since is None:
            since = 0
        start = int((since - 1700000000000) / 900000)
        start = max(start, 0)
        return self._rows[start:start + 1000]


def test_cache_path_sanitizes_symbol():
    p = cache_path("BTC/USDT:USDT", "15m")
    assert ":" not in p.name and "/" not in p.name


def test_fetch_ohlcv_long_paginates(tmp_path):
    fake = FakeBinance()
    df = fetch_ohlcv_long(fake, "BTC/USDT:USDT", "15m", limit=1200,
                          cache_dir=tmp_path)
    assert len(df) == 1200
    assert df["close"].iloc[-1] == 100.5
    assert isinstance(df.index, pd.DatetimeIndex)


def test_fetch_ohlcv_long_uses_cache(tmp_path):
    fake = FakeBinance()
    df1 = fetch_ohlcv_long(fake, "BTC/USDT:USDT", "15m", limit=1200,
                           cache_dir=tmp_path)
    fake._rows = []  # if cache used, no refetch -> still returns data
    df2 = fetch_ohlcv_long(fake, "BTC/USDT:USDT", "15m", limit=1200,
                           cache_dir=tmp_path)
    assert len(df2) == 1200
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd <worktree> && /Users/unknown965/coding/OpenCodeTest/crypto-invest-analysis/.venv/bin/python -m pytest tests/test_binance_data.py -q`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Write minimal implementation**

```python
# bot/binance_data.py
"""Long-history OHLCV fetch from Binance public API + CSV cache.

Calibration-only. The live bot still executes on OKX demo.
"""
import pandas as pd
from pathlib import Path

DEFAULT_CACHE_DIR = Path(__file__).parent / "cache_binance"
BATCH = 1000


def cache_path(symbol: str, timeframe: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    safe = symbol.replace("/", "_").replace(":", "_")
    return cache_dir / f"{safe}_{timeframe}.csv"


def fetch_ohlcv_long(exchange, symbol: str, timeframe: str, limit: int,
                     cache_dir: Path = DEFAULT_CACHE_DIR, force_refresh: bool = False) -> pd.DataFrame:
    path = cache_path(symbol, timeframe, cache_dir)
    if path.exists() and not force_refresh:
        df = pd.read_csv(path, index_col="timestamp", parse_dates=True)
        if len(df) >= limit:
            return df.tail(limit)

    interval_ms = {"5m": 300000, "15m": 900000, "1h": 3600000}[timeframe]
    since = int(pd.Timestamp.now().timestamp() * 1000) - limit * interval_ms

    rows = []
    cursor = since
    while len(rows) < limit:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=BATCH)
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < BATCH:
            break
        cursor = batch[-1][0] + 1
        if len(rows) >= limit:
            break

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")
    df = df.set_index("timestamp").tail(limit)

    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return df


def get_binance_exchange():
    import ccxt
    return ccxt.binance({"enableRateLimit": True})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_binance_data.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add bot/binance_data.py tests/test_binance_data.py
git commit -m "feat(bot): Binance long-history fetcher with CSV cache"
```

---

### Task 2: bot/risk.py — ATR trailing stop

**Files:**
- Modify: `bot/risk.py`
- Test: `tests/test_bot_risk.py`

- [ ] **Step 1: Add failing tests**

```python
# append to tests/test_bot_risk.py
from bot.risk import trailing_stop_price, check_trailing_stop


def test_trailing_stop_price_long_uses_peak():
    # long: peak 110, ATR 2, k 2.5 -> atr_stop = 110 - 5 = 105; hard_stop = 100*0.975 = 97.5
    stop = trailing_stop_price(100.0, 110.0, 2.0, "long", 2.5, 0.025)
    assert stop == 105.0  # max(105, 97.5)


def test_trailing_stop_price_long_floor_hard_stop():
    # low peak so ATR stop below hard stop -> hard stop wins
    stop = trailing_stop_price(100.0, 100.0, 2.0, "long", 2.5, 0.025)
    assert stop == 97.5


def test_trailing_stop_price_short_uses_peak():
    stop = trailing_stop_price(100.0, 90.0, 2.0, "short", 2.5, 0.025)
    # atr_stop = 90 + 5 = 95; hard_stop = 102.5; min(95, 102.5) = 95
    assert stop == 95.0


def test_check_trailing_stop_long_fires_below_stop():
    assert check_trailing_stop(100.0, 110.0, 104.5, 2.0, "long", 2.5, 0.025) is True
    assert check_trailing_stop(100.0, 110.0, 105.5, 2.0, "long", 2.5, 0.025) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_bot_risk.py -q`
Expected: FAIL (`ImportError: cannot import name 'trailing_stop_price'`)

- [ ] **Step 3: Add implementation**

```python
# append to bot/risk.py
def trailing_stop_price(entry_price: float, peak_price: float, atr: float,
                        side: str, k: float, hard_stop_pct: float) -> float:
    """Current stop price for a trailing stop (hard stop acts as floor/ceiling)."""
    if side == "long":
        atr_stop = peak_price - k * atr
        hard_stop = entry_price * (1 - hard_stop_pct)
        return max(atr_stop, hard_stop)
    atr_stop = peak_price + k * atr
    hard_stop = entry_price * (1 + hard_stop_pct)
    return min(atr_stop, hard_stop)


def check_trailing_stop(entry_price: float, peak_price: float, current_price: float,
                        atr: float, side: str, k: float, hard_stop_pct: float) -> bool:
    stop = trailing_stop_price(entry_price, peak_price, atr, side, k, hard_stop_pct)
    if side == "long":
        return current_price <= stop
    return current_price >= stop
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_bot_risk.py -q`
Expected: PASS (10 passed)

- [ ] **Step 5: Commit**

```bash
git add bot/risk.py tests/test_bot_risk.py
git commit -m "feat(bot): ATR trailing stop with hard-stop floor"
```

---

### Task 3: bot/signals.py — volatility filter + latest_atr

**Files:**
- Modify: `bot/signals.py`
- Test: `tests/test_bot_signals.py`

- [ ] **Step 1: Add failing tests**

```python
# append to tests/test_bot_signals.py
from bot.signals import volatility_ok, latest_atr


def test_latest_atr_returns_positive():
    df = _ohlcv(300, trend=0.0005)
    assert latest_atr(df) > 0


def test_volatility_ok_high_vol():
    df = _ohlcv(300, trend=0.0005)
    assert volatility_ok(df, min_atr_pct=0.0001) is True


def test_volatility_ok_low_vol_blocked():
    idx = pd.date_range("2026-01-01", periods=300, freq="5min")
    close = pd.Series([40000.0] * 300, index=idx)  # perfectly flat -> ATR ~0
    df = pd.DataFrame({"open": close, "high": close, "low": close,
                       "close": close, "volume": 100.0}, index=idx)
    assert volatility_ok(df, min_atr_pct=0.0001) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_bot_signals.py -q`
Expected: FAIL (`ImportError`)

- [ ] **Step 3: Add implementation**

```python
# append to bot/signals.py
def latest_atr(df: pd.DataFrame) -> float:
    ind = compute_all_indicators(df)
    val = ind["ATR"].iloc[-1]
    return float(val) if pd.notna(val) else 0.0


def volatility_ok(df_ltf: pd.DataFrame, min_atr_pct: float) -> bool:
    atr = latest_atr(df_ltf)
    close = float(df_ltf["close"].iloc[-1])
    if close <= 0:
        return False
    return atr / close >= min_atr_pct
```

(ensure `import pandas as pd` is present in signals.py — it already is)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_bot_signals.py -q`
Expected: PASS (10 passed)

- [ ] **Step 5: Commit**

```bash
git add bot/signals.py tests/test_bot_signals.py
git commit -m "feat(bot): volatility filter + latest ATR helper"
```

---

### Task 4: bot/config.py — new params

**Files:**
- Modify: `bot/config.py`
- Test: `tests/test_bot_config.py`

- [ ] **Step 1: Add failing test**

```python
# append to tests/test_bot_config.py
def test_signal_v2_defaults():
    cfg = BotConfig()
    assert cfg.atr_stop_mult == 2.5
    assert cfg.min_atr_pct == 0.0015
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL (`AttributeError`)

- [ ] **Step 3: Add fields**

```python
    atr_stop_mult: float = 2.5
    min_atr_pct: float = 0.0015
```

(add after `stop_loss_pct` in the dataclass)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_bot_config.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add bot/config.py tests/test_bot_config.py
git commit -m "feat(bot): ATR stop mult + min volatility config"
```

---

### Task 5: bot/loop.py — trailing stop, vol filter, peak tracking

**Files:**
- Modify: `bot/loop.py`
- Test: `tests/test_bot_loop.py`

- [ ] **Step 1: Add failing tests**

```python
# append to tests/test_bot_loop.py
from bot.risk import check_trailing_stop  # existing import pattern
from bot.signals import volatility_ok     # ensure importable


def test_loop_tracks_peak_and_trailing_stop(tmp_path):
    # V-bull entry then a price drop after peak -> trailing stop fires
    logs = []
    cfg = BotConfig(dry_run=True, poll_seconds=0, symbols=["BTC/USDT:USDT"],
                    atr_stop_mult=0.05, min_atr_pct=0.0)  # tiny k so stop is tight
    calls = {"n": 0}

    def series(trend, base):
        idx = pd.date_range("2026-01-01", periods=300, freq="5min")
        close = base * (1 + trend) ** np.arange(300)
        return pd.DataFrame({"open": close*0.999, "high": close*1.002, "low": close*0.998,
                             "close": close, "volume": 100.0}, index=idx)

    def fetcher(symbol, timeframe, limit):
        calls["n"] += 1
        if timeframe == "1h":
            return series(0.0005, 40000.0)
        # LTF: enter in a V (up-cross), then crash after entry
        n = calls["n"]
        if n <= 2:
            return series(0.0, 40000.0)
        return series(-0.001, 40000.0)  # crashing -> price < trailing stop

    run_bot(cfg, executor=None, fetch_fn=fetcher, state_path=tmp_path / "s.json",
            max_iterations=6, logger=logs.append)
    state = __import__("bot.state", fromlist=["load_state"]).load_state(tmp_path / "s.json")
    assert any("trailing stop" in line or "stop loss" in line for line in logs)


def test_loop_vol_filter_blocks_low_vol_entry(tmp_path):
    logs = []
    cfg = BotConfig(dry_run=True, poll_seconds=0, symbols=["BTC/USDT:USDT"], min_atr_pct=10.0)
    # min_atr_pct huge -> always blocked
    fetcher = FakeFetcher(htf_trend=0.0005, ltf_trend=0.0005)
    run_bot(cfg, executor=None, fetch_fn=fetcher, state_path=tmp_path / "s.json",
            max_iterations=5, logger=logs.append)
    state = __import__("bot.state", fromlist=["load_state"]).load_state(tmp_path / "s.json")
    assert state["positions"]["BTC/USDT:USDT"] is None
    assert any("volatility" in line for line in logs)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_bot_loop.py -q`
Expected: FAIL (trailing-stop test won't fire; vol-filter test shows position opened)

- [ ] **Step 3: Modify `bot/loop.py`**

In `_tick`:

```python
    # per-trade stop: trailing stop (with ATR) + hard-stop floor
    if pos is not None:
        # update peak first
        if pos["side"] == "long":
            pos["peak"] = max(pos.get("peak", pos["entry_price"]), price)
        else:
            pos["peak"] = min(pos.get("peak", pos["entry_price"]), price)
        atr = latest_atr(df_ltf)
        if check_trailing_stop(pos["entry_price"], pos["peak"], price, atr,
                               pos["side"], cfg.atr_stop_mult, cfg.stop_loss_pct):
            logger(f"[{symbol}] trailing stop hit @ {price:.2f}")
            _realize(state, symbol, pos, price)
            if not cfg.dry_run:
                executor.close_position(symbol, pos["side"])
            return
```

In `_tick`, entry path — gate on volatility before evaluating entry:

```python
    # volatility filter (entries only)
    if pos is None and not volatility_ok(df_ltf, cfg.min_atr_pct):
        logger(f"[{symbol}] volatility below threshold, skip entry")
        return
```

And when opening a position, initialize `peak`:

```python
    state["positions"][symbol] = {"side": ..., "entry_price": price, "qty": qty,
                                  "entry_time": ..., "peak": price}
```

(update BOTH dry-run and live branches)

Also update imports at top of loop.py:
```python
from bot.signals import evaluate, latest_atr, volatility_ok
from bot.risk import (check_daily_loss, check_trailing_stop,
                      check_position_limit, position_notional)
```
(remove the now-unused `check_stop_loss` import if it becomes unused — check first)

IMPORTANT: `df_ltf` is available in `_tick` (fetched at top). `latest_atr`/`volatility_ok` need `df_ltf`. Verify placement — the stop check uses `df_ltf`, the vol filter uses `df_ltf`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_bot_loop.py -q`
Expected: PASS (existing + 2 new). If the trailing-stop test is flaky with synthetic ATR values, tune the `atr_stop_mult` in the test (tiny k → tight stop) until deterministic. If the existing `test_dry_run_opens_virtual_long` now fails because the vol filter blocks entry (its synthetic series may have near-zero ATR), set `min_atr_pct=0.0` in the cfg for that test so it passes vol filter.

- [ ] **Step 5: Run full suite + commit**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all green.

```bash
git add bot/loop.py tests/test_bot_loop.py
git commit -m "feat(bot): ATR trailing stop + volatility filter in loop"
```

---

### Task 6: bot/backtest_calibration.py — Binance data + new param grid

**Files:**
- Modify: `bot/backtest_calibration.py`

- [ ] **Step 1: Rewrite the script**

```python
# bot/backtest_calibration.py
"""M1 v2: Backtest hybrid signal on Binance long history; calibrate params.

Grid: htf_threshold x stop mode (hard/trailing) x atr_stop_mult x min_atr_pct.
Requires network (Binance public API). Saves bot/calibration_results_v2.json.
"""
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.binance_data import fetch_ohlcv_long, get_binance_exchange
from bot.signals import momentum_series, latest_atr, volatility_ok
from bot.risk import check_trailing_stop

SYMBOLS = ["BTC/USDT:USDT", "SOL/USDT:USDT"]
HTF, LTF = "1h", "15m"
HTF_LIMIT, LTF_LIMIT = 800, 8000   # ~33 days HTF, ~83 days LTF
FEE_RATE = 0.001
HARD_STOP_PCT = 0.025
INITIAL = 10000.0

GRID = {
    "threshold": [0.10, 0.15, 0.20, 0.25, 0.30],
    "trailing": [False, True],
    "atr_mult": [2.0, 3.0, 4.0],
    "min_atr_pct": [0.0, 0.001, 0.002],
}


def run_backtest(df_htf, df_ltf, threshold, trailing, atr_mult, min_atr_pct,
                 fee_rate=FEE_RATE, initial=INITIAL):
    htf_score = momentum_series(df_htf)
    ltf_score = momentum_series(df_ltf)
    ltf_delta = ltf_score.diff()
    atr = latest_atr_series(df_ltf)

    ltf = df_ltf.copy()
    ltf["htf_score"] = htf_score.reindex(ltf.index, method="ffill")
    ltf["delta"] = ltf_delta.reindex(ltf.index)
    ltf["delta_prev"] = ltf["delta"].shift(1)
    ltf["atr"] = atr.reindex(ltf.index)
    ltf["vol_ok"] = (ltf["atr"] / ltf["close"]) >= min_atr_pct
    ltf = ltf.dropna(subset=["htf_score", "delta", "delta_prev", "atr"])
    if len(ltf) == 0:
        return {"return_pct": 0.0, "trades": 0, "max_dd_pct": 0.0}

    equity = initial
    position = None
    entry_price = peak = None
    trades = 0
    curve_peak = initial
    max_dd = 0.0

    for _, row in ltf.iterrows():
        price, direction = row["close"], (
            "long" if row["htf_score"] > threshold
            else "short" if row["htf_score"] < -threshold else "flat")
        up, down = row["delta_prev"] <= 0 < row["delta"], row["delta_prev"] >= 0 > row["delta"]

        if position is None:
            if row["vol_ok"] and direction == "long" and up:
                position, entry_price, peak = "long", price, price
                trades += 1
            elif row["vol_ok"] and direction == "short" and down:
                position, entry_price, peak = "short", price, price
                trades += 1
        else:
            if position == "long":
                peak = max(peak, price)
                stop_hit = check_trailing_stop(entry_price, peak, price, row["atr"],
                                               "long", atr_mult if trailing else 0.0,
                                               HARD_STOP_PCT)
            else:
                peak = min(peak, price)
                stop_hit = check_trailing_stop(entry_price, peak, price, row["atr"],
                                               "short", atr_mult if trailing else 0.0,
                                               HARD_STOP_PCT)
            exit_now = stop_hit or (
                (position == "long" and (direction != "long" or down))
                or (position == "short" and (direction != "short" or up)))
            if exit_now:
                ret = (price / entry_price - 1) * (1 if position == "long" else -1)
                equity *= 1 + ret - 2 * fee_rate
                position = entry_price = peak = None

        curve_peak = max(curve_peak, equity)
        max_dd = max(max_dd, (curve_peak - equity) / curve_peak)

    return {"return_pct": (equity / initial - 1) * 100, "trades": trades,
            "max_dd_pct": max_dd * 100}


def latest_atr_series(df):
    import pandas as pd
    from feature_engine.indicators import compute_all_indicators
    return compute_all_indicators(df)["ATR"]


def calibrate():
    ex = get_binance_exchange()
    out = {}
    for symbol in SYMBOLS:
        print(f"=== {symbol} ===")
        df_htf = fetch_ohlcv_long(ex, symbol, HTF, HTF_LIMIT)
        df_ltf = fetch_ohlcv_long(ex, symbol, LTF, LTF_LIMIT)
        print(f"  1h: {len(df_htf)} bars {df_htf.index[0]} -> {df_htf.index[-1]}")
        print(f"  15m: {len(df_ltf)} bars {df_ltf.index[0]} -> {df_ltf.index[-1]}")

        best = None
        n = 0
        for threshold, trailing, mult, atr_pct in itertools.product(
                GRID["threshold"], GRID["trailing"], GRID["atr_mult"], GRID["min_atr_pct"]):
            r = run_backtest(df_htf, df_ltf, threshold, trailing, mult, atr_pct)
            n += 1
            if best is None or r["return_pct"] > best[1]["return_pct"]:
                best = ((threshold, trailing, mult, atr_pct), r)
        print(f"  ({n} combos tested)")
        print(f"  BEST: thr={best[0][0]} trailing={best[0][1]} mult={best[0][2]} "
              f"atr_pct={best[0][3]} -> ret {best[1]['return_pct']:+.1f}% "
              f"trades={best[1]['trades']} maxDD {best[1]['max_dd_pct']:.1f}%")
        out[symbol] = {"params": best[0], **best[1]}

    with open(Path(__file__).parent / "calibration_results_v2.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved bot/calibration_results_v2.json")
    return out


if __name__ == "__main__":
    calibrate()
```

Note: `check_trailing_stop(entry, peak, price, atr, side, k=0.0, hard)` with k=0.0 → atr_stop = peak, and hard stop floor still applies — this implements the plain hard-stop case when `trailing=False`. Verify this equivalence holds (peak±0 = peak; long: max(peak, hard_stop) → hard stop fires at entry_price*(1-0.025); correct for pure hard stop).

- [ ] **Step 2: Run the calibration**

Run: `cd <worktree> && .venv/bin/python bot/backtest_calibration.py`
Expected: fetches Binance long history for both symbols (may take 1-3 min), prints per-symbol best config, saves `bot/calibration_results_v2.json`.

**Fail-loud gate:** If the best config for EITHER symbol has `trades < 100` OR `return_pct <= 0`, this is a negative result. Do NOT skip it — include it clearly in your report.

- [ ] **Step 3: Commit**

```bash
git add bot/backtest_calibration.py bot/calibration_results_v2.json
git commit -m "feat(bot): v2 calibration on Binance data (ATR stop + vol filter grid)"
```

---

### Task 7: Apply calibrated params to config + README

**Files:**
- Modify: `bot/config.py`, `README.md`

- [ ] **Step 1: Read `bot/calibration_results_v2.json`** and update `bot/config.py`:
   - If positive result: set `htf_threshold`, `atr_stop_mult`, `min_atr_pct` to the best per-symbol values (use average rounded to grid step if they differ), with a comment citing the JSON.
   - If negative result: keep v1 defaults, update the README with the honest v2 finding (no edge demonstrated even with 6+ months Binance data).
- [ ] **Step 2: Update README HFT section** with v2 calibration outcome (best params + return/trades, or the negative finding).
- [ ] **Step 3: Run full suite** `.venv/bin/python -m pytest tests/ -q` (green).
- [ ] **Step 4: Commit**

```bash
git add bot/config.py README.md
git commit -m "docs(bot): v2 calibration outcome in config + README"
```

---

## Self-Review

**Spec coverage:**
- Binance long history → Task 1, 6 ✅
- ATR trailing stop → Task 2 (risk), Task 5 (loop), Task 6 (backtest) ✅
- Volatility filter → Task 3 (signals), Task 5 (loop), Task 6 (backtest) ✅
- state peak field → Task 5 (positions store peak; state.py unchanged, dict is flexible) ✅
- config params → Task 4 ✅
- Re-calibration + validation gate → Task 6, 7 ✅

**Type consistency:**
- `trailing_stop_price(entry, peak, atr, side, k, hard_stop_pct)` / `check_trailing_stop(entry, peak, price, atr, side, k, hard_stop_pct)` — consistent across Task 2 (risk), Task 5 (loop), Task 6 (backtest). Loop passes `(entry, peak, price, atr, side, cfg.atr_stop_mult, cfg.stop_loss_pct)`. ✅
- `volatility_ok(df, min_atr_pct)` consistent Task 3, 5, 6. ✅
- `latest_atr(df)` returns float; backtest uses `latest_atr_series(df)` (pandas Series) — DIFFERENT function, name is distinct so no collision; backtest file defines its own series version. ✅
- Config fields `atr_stop_mult`, `min_atr_pct` consistent Task 4, 5. ✅
- `fetch_ohlcv_long(exchange, symbol, timeframe, limit, cache_dir, force_refresh)` consistent Task 1, 6. ✅

**Known test-flakiness risks (documented for the executor):** Task 5 trailing-stop test may need `atr_stop_mult` tuning; existing loop test `test_dry_run_opens_virtual_long` may need `min_atr_pct=0.0` to pass the new vol filter. The plan instructs the implementer how to handle both.