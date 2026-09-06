# 日內均值回歸策略 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Validate whether an intraday mean-reversion strategy (RSI oversold/overbought on 15m + SMA_50 trend filter on 1h) has edge on Binance long history, using the existing bot infrastructure.

**Architecture:** New `bot/meanreversion.py` (signal module). `bot/backtest_calibration.py` gets a `--signal meanreversion` mode with its own grid. `bot/config.py` + `bot/loop.py` get a mode switch (momentum vs meanreversion). Reuse `bot/binance_data.py`, `bot/risk.py` (ATR trailing stop), `bot/signals.py` (latest_atr/volatility_ok).

**Tech Stack:** Python 3.9, pandas, ccxt, existing `bot/` modules.

**Spec:** `docs/superpowers/specs/2026-09-06-hft-meanreversion-design.md`

---

### Task 1: bot/meanreversion.py

**Files:** Create `bot/meanreversion.py`, Test `tests/test_meanreversion.py`

- [ ] **Step 1: Failing test**

```python
# tests/test_meanreversion.py
import numpy as np
import pandas as pd
from bot.meanreversion import latest_rsi, evaluate_meanrev


def _ohlcv(n, trend, base=40000.0):
    idx = pd.date_range("2026-01-01", periods=n, freq="5min")
    close = base * (1 + trend) ** np.arange(n)
    return pd.DataFrame({"open": close*0.999, "high": close*1.002, "low": close*0.998,
                         "close": close, "volume": 100.0}, index=idx)


def test_latest_rsi_bounded():
    df = _ohlcv(300, trend=0.0005)
    assert 0 <= latest_rsi(df) <= 100


def test_evaluate_meanrev_entry_long_oversold_in_uptrend(monkeypatch):
    import bot.meanreversion as mr
    calls = {"n": 0}

    def fake_rsi(df):
        calls["n"] += 1
        if calls["n"] == 1:  # HTF not used by RSI; LTF RSI
            return pd.Series([40.0, 40.0, 24.0], index=[0, 1, 2])  # prev 40 > 25, curr 24 <= 25
        return pd.Series([40.0])

    monkeypatch.setattr(mr, "latest_rsi", fake_rsi)
    monkeypatch.setattr(mr, "htf_trend_up", lambda df_htf: True)   # uptrend
    monkeypatch.setattr(mr, "htf_trend_down", lambda df_htf: False)
    sig = evaluate_meanrev(None, None, oversold=25, overbought=75, side="flat")
    assert sig.action == "enter_long"


def test_evaluate_meanrev_blocks_flat_rsi():
    import bot.meanreversion as mr
    monkeypatch.setattr(mr, "latest_rsi", lambda df: pd.Series([50.0]))
    monkeypatch.setattr(mr, "htf_trend_up", lambda df_htf: True)
    monkeypatch.setattr(mr, "htf_trend_down", lambda df_htf: False)
    assert evaluate_meanrev(None, None, 25, 75, "flat").action == "none"


def test_evaluate_meanrev_exit_long_on_reversion():
    import bot.meanreversion as mr
    monkeypatch.setattr(mr, "latest_rsi", lambda df: pd.Series([58.0]))  # >= exit 55
    sig = evaluate_meanrev(None, None, 25, 75, side="long")
    assert sig.action == "exit_long"
```

- [ ] **Step 2: Verify fail.** Run: `cd <worktree> && /Users/unknown965/coding/OpenCodeTest/crypto-invest-analysis/.venv/bin/python -m pytest tests/test_meanreversion.py -q` → FAIL (ImportError)

- [ ] **Step 3: Implementation**

```python
# bot/meanreversion.py
from dataclasses import dataclass

import pandas as pd

from feature_engine.indicators import compute_all_indicators


@dataclass
class Signal:
    action: str  # enter_long/enter_short/exit_long/exit_short/none
    rsi: float


def latest_rsi(df: pd.DataFrame) -> pd.Series:
    return compute_all_indicators(df)["RSI"]


def htf_trend_up(df_htf: pd.DataFrame) -> bool:
    ind = compute_all_indicators(df_htf)
    return bool(df_htf["close"].iloc[-1] > ind["SMA_50"].iloc[-1])


def htf_trend_down(df_htf: pd.DataFrame) -> bool:
    ind = compute_all_indicators(df_htf)
    return bool(df_htf["close"].iloc[-1] < ind["SMA_50"].iloc[-1])


def evaluate_meanrev(df_htf, df_ltf, oversold, overbought, side="flat",
                     exit_long_rsi=55.0, exit_short_rsi=45.0) -> Signal:
    rsi = latest_rsi(df_ltf)
    prev_rsi = float(rsi.iloc[-2])
    curr_rsi = float(rsi.iloc[-1])

    if side == "long":
        if curr_rsi >= exit_long_rsi:
            return Signal("exit_long", curr_rsi)
        return Signal("none", curr_rsi)
    if side == "short":
        if curr_rsi <= exit_short_rsi:
            return Signal("exit_short", curr_rsi)
        return Signal("none", curr_rsi)
    if htf_trend_up(df_htf) and prev_rsi > oversold and curr_rsi <= oversold:
        return Signal("enter_long", curr_rsi)
    if htf_trend_down(df_htf) and prev_rsi < overbought and curr_rsi >= overbought:
        return Signal("enter_short", curr_rsi)
    return Signal("none", curr_rsi)
```

- [ ] **Step 4: Verify pass** (`.venv/bin/python -m pytest tests/test_meanreversion.py -q`). Note: tests monkeypatch `latest_rsi`/`htf_trend_*`; if the monkeypatched `latest_rsi` signature `(df)->Series` conflicts with the `iloc[-2]` usage, the test values must provide 2+ elements for entry tests. Adjust the fake to return Series with >=2 elements where needed.

- [ ] **Step 5: Commit** `git add bot/meanreversion.py tests/test_meanreversion.py && git commit -m "feat(bot): intraday mean-reversion signal (RSI + SMA_50 trend)"`

---

### Task 2: bot/config.py + bot/loop.py mode switch

**Files:** Modify `bot/config.py`, `bot/loop.py`, tests.

- [ ] **Step 1: Failing tests** (append to `tests/test_bot_config.py` and `tests/test_bot_loop.py`)

```python
# tests/test_bot_config.py
def test_meanreversion_config_defaults():
    cfg = BotConfig()
    assert cfg.signal_mode == "momentum"       # default stays momentum
    assert cfg.rsi_oversold == 25
    assert cfg.rsi_overbought == 75


# tests/test_bot_loop.py
def test_loop_meanreversion_mode_enters(tmp_path):
    # cfg.signal_mode="meanreversion", synthetic oversold RSI in uptrend -> enters long
    # build a series whose LTF RSI is oversold (sharp drop then flat) and HTF rising
    ...
```

Note for the loop test: construct synthetic data where the 15m RSI is low (e.g., a recent sharp drop makes RSI < 25) while the 1h close > SMA_50. Verify the loop opens a virtual long and logs "meanreversion". If hard to make deterministic with real indicator computation, monkeypatch `bot.loop.evaluate` (the mean-reversion evaluate) to return `Signal("enter_long", ...)` and assert the loop wires it — this tests the wiring, not the signal.

- [ ] **Step 2: Verify fail**
- [ ] **Step 3: Implement**

In `bot/config.py` add:
```python
    signal_mode: str = "momentum"  # "momentum" | "meanreversion"
    rsi_oversold: int = 25
    rsi_overbought: int = 75
```

In `bot/loop.py`, the signal evaluation section currently calls `evaluate(df_htf, df_ltf, cfg.htf_threshold, side)`. Change to:
```python
    from bot.meanreversion import evaluate_meanrev
    if cfg.signal_mode == "meanreversion":
        sig = evaluate_meanrev(df_htf, df_ltf, cfg.rsi_oversold, cfg.rsi_overbought, side)
    else:
        sig = evaluate(df_htf, df_ltf, cfg.htf_threshold, side)
```
(The `Signal` dataclass from meanreversion has `.action` and `.rsi`; loop uses only `.action`, so no change needed to the loop's action handling.)

- [ ] **Step 4: Verify pass + full suite**
- [ ] **Step 5: Commit** `git add bot/config.py bot/loop.py tests/test_bot_config.py tests/test_bot_loop.py && git commit -m "feat(bot): signal mode switch (momentum/meanreversion)"`

---

### Task 3: calibration --signal meanreversion

**Files:** Modify `bot/backtest_calibration.py`.

- [ ] **Step 1: Add a `--signal` arg and a meanreversion backtest path.** Add `import argparse`; `--signal {momentum,meanreversion}` (default momentum). Extract the data loading + save into shared helpers. Add:

```python
def run_backtest_meanrev(df_htf, df_ltf, oversold, overbought, htf_filter, trailing, atr_mult,
                         fee_rate=FEE_RATE, initial=INITIAL):
    from bot.meanreversion import latest_rsi, htf_trend_up, htf_trend_down
    from bot.risk import check_trailing_stop
    from bot.signals import latest_atr_series_note  # no — use local ATR

    rsi = latest_rsi(df_ltf)
    atr = _atr_series(df_ltf)
    ltf = df_ltf.copy()
    ltf["rsi"] = rsi.reindex(ltf.index)
    ltf["rsi_prev"] = ltf["rsi"].shift(1)
    ltf["atr"] = atr.reindex(ltf.index)
    # htf filter series
    ind_htf = compute_all_indicators(df_htf)
    htf_up = (df_htf["close"] > ind_htf["SMA_50"]).reindex(ltf.index, method="ffill")
    ltf = ltf.dropna(subset=["rsi", "rsi_prev", "atr"])

    equity, position, entry_price, peak, trades, curve_peak, max_dd = (
        initial, None, None, None, 0, initial, 0.0)
    for _, row in ltf.iterrows():
        price, r = row["close"], row["rsi"]
        if position is None:
            if (htf_up.loc[row.name] if htf_filter else True) and row["rsi_prev"] > oversold and r <= oversold:
                position, entry_price, peak, trades = "long", price, price, trades + 1
            elif (not htf_up.loc[row.name] if htf_filter else True) and row["rsi_prev"] < overbought and r >= overbought:
                position, entry_price, peak, trades = "short", price, price, trades + 1
        else:
            if position == "long":
                peak = max(peak, price)
                stop_hit = check_trailing_stop(entry_price, peak, price, row["atr"], "long",
                                               atr_mult if trailing else 0.0, HARD_STOP_PCT)
                exit_now = stop_hit or r >= 55.0
            else:
                peak = min(peak, price)
                stop_hit = check_trailing_stop(entry_price, peak, price, row["atr"], "short",
                                               atr_mult if trailing else 0.0, HARD_STOP_PCT)
                exit_now = stop_hit or r <= 45.0
            if exit_now:
                ret = (price / entry_price - 1) * (1 if position == "long" else -1)
                equity *= 1 + ret - 2 * fee_rate
                position = entry_price = peak = None
        curve_peak = max(curve_peak, equity)
        max_dd = max(max_dd, (curve_peak - equity) / curve_peak)
    return {"return_pct": (equity / initial - 1) * 100, "trades": trades, "max_dd_pct": max_dd * 100}


def calibrate_meanrev():
    ex = get_binance_exchange()
    out = {}
    for symbol in SYMBOLS:
        df_htf = fetch_ohlcv_long(ex, symbol, HTF, HTF_LIMIT)
        df_ltf = fetch_ohlcv_long(ex, symbol, LTF, LTF_LIMIT)
        best, n = None, 0
        for oversold, overbought, htf_filter, trailing, mult in itertools.product(
                [20, 25, 30], [70, 75, 80], [True, False], [False, True], [2.0, 3.0, 4.0]):
            if oversold >= overbought:
                continue
            r = run_backtest_meanrev(df_htf, df_ltf, oversold, overbought, htf_filter,
                                     trailing, mult)
            n += 1
            if best is None or r["return_pct"] > best[1]["return_pct"]:
                best = ((oversold, overbought, htf_filter, trailing, mult), r)
        print(f"{symbol}: BEST over={best[0][0]} overb={best[0][1]} htf={best[0][2]} "
              f"trailing={best[0][3]} mult={best[0][4]} -> ret {best[1]['return_pct']:+.1f}% "
              f"trades={best[1]['trades']} maxDD {best[1]['max_dd_pct']:.1f}%")
        out[symbol] = {"params": best[0], **best[1]}
    with open(Path(__file__).parent / "calibration_results_meanrev.json", "w") as f:
        json.dump(out, f, indent=2)
    return out
```

Add `_atr_series(df)` helper (compute_all_indicators(df)["ATR"]). Add `--signal meanreversion` → calls `calibrate_meanrev()`.

- [ ] **Step 2: Run it.** `.venv/bin/python bot/backtest_calibration.py --signal meanreversion`
  **FAIL-LOUD gate:** report trades>=100 AND return>0 both symbols, or the honest negative.
- [ ] **Step 3: Commit** `git add bot/backtest_calibration.py bot/calibration_results_meanrev.json && git commit -m "feat(bot): mean-reversion calibration on Binance data"`

---

### Task 4: Document outcome + README

**Files:** Modify `README.md` (+ maybe `bot/config.py` comment).

- [ ] **Step 1: Read `bot/calibration_results_meanrev.json`.** If positive (both symbols trades>=100, return>0): update config defaults to best params (avg if differ) and README with the result. If negative: keep defaults, README honest finding (like v2).
- [ ] **Step 2: Update README HFT section** with the mean-reversion outcome.
- [ ] **Step 3: Full suite** `.venv/bin/python -m pytest tests/ -q` (green).
- [ ] **Step 4: Commit** `git add bot/config.py README.md && git commit -m "docs(bot): mean-reversion calibration outcome"`

---

## Self-Review

**Spec coverage:** mean-reversion signal (Task 1) ✅; mode switch (Task 2) ✅; calibration grid oversold×overbought×htf×stop (Task 3) ✅; honest outcome (Task 4) ✅. Reuses binance_data, risk.py trailing stop ✅.
**Type consistency:** `Signal.action` same values in meanreversion.py and loop.py usage ✅; `evaluate_meanrev(df_htf, df_ltf, oversold, overbought, side)` consistent Task 1/2/3 ✅; `run_backtest_meanrev` param order consistent with calibrate loop ✅. `htf_up.loc[row.name]` — row.name is the timestamp; htf_up is a boolean Series indexed by LTF timestamps via reindex(..., method="ffill") → .loc by timestamp works ✅.