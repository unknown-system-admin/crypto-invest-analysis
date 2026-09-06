# 資金費率濾網 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`).

**Goal:** Validate whether funding-rate-as-contrarian-filter adds edge to the v3 mean-reversion signal. Calibrate on long funding history. Document honestly.

**Architecture:** New `bot/funding.py` (funding history + bias). `bot/meanreversion.py` gains optional funding bias in `evaluate_meanrev`. `bot/backtest_calibration.py` `--signal meanreversion` adds funding-threshold to the grid. `bot/config.py`/`bot/loop.py` wiring.

**Tech Stack:** Python 3.9, ccxt (binance/okx funding history), existing bot modules.

**Spec:** `docs/superpowers/specs/2026-09-06-hft-funding-filter-design.md`

---

### Task 1: bot/funding.py

**Files:** Create `bot/funding.py`, Test `tests/test_funding.py`

- [ ] **Step 1: Failing test**

```python
# tests/test_funding.py
import pandas as pd
from bot.funding import funding_bias, funding_series_from_rows


def test_funding_bias_positive_threshold():
    assert funding_bias(0.0005, 0.0001) == "short"    # crowded longs -> short
    assert funding_bias(-0.0005, 0.0001) == "long"    # crowded shorts -> long
    assert funding_bias(0.00005, 0.0001) == "neutral"  # mild -> no bias


def test_funding_series_from_rows_sorts_and_indexes():
    rows = [
        {"timestamp": 2000, "fundingRate": 0.0002},
        {"timestamp": 1000, "fundingRate": 0.0001},
        {"timestamp": 3000, "fundingRate": -0.0001},
    ]
    s = funding_series_from_rows(rows)
    assert isinstance(s, pd.Series)
    assert s.index.is_monotonic_increasing
    assert float(s.iloc[0]) == 0.0001
    assert float(s.iloc[-1]) == -0.0001


def test_funding_bias_works_with_series_value():
    # threshold on a single float value pulled from a Series
    s = pd.Series([0.0005])
    assert funding_bias(float(s.iloc[-1]), 0.0001) == "short"
```

- [ ] **Step 2: Verify fail** `.venv/bin/python -m pytest tests/test_funding.py -q` → ImportError
- [ ] **Step 3: Implementation**

```python
# bot/funding.py
"""Funding-rate contrarian bias + history helpers."""
import pandas as pd


def funding_bias(rate: float, threshold: float) -> str:
    """Contrarian funding bias. Positive funding = crowded longs -> short bias."""
    if rate > threshold:
        return "short"
    if rate < -threshold:
        return "long"
    return "neutral"


def funding_series_from_rows(rows: list) -> pd.Series:
    """Build a sorted funding-rate Series indexed by timestamp (ms)."""
    if not rows:
        return pd.Series(dtype=float)
    ts = pd.to_datetime([r["timestamp"] for r in rows], unit="ms")
    rates = [r.get("fundingRate") for r in rows]
    s = pd.Series(rates, index=ts)
    return s[~s.index.duplicated(keep="last")].sort_index().astype(float)


def fetch_funding_history(exchange, symbol: str) -> pd.Series:
    """Fetch funding history; return sorted Series indexed by timestamp."""
    rows = exchange.fetch_funding_rate_history(symbol)
    return funding_series_from_rows(rows)


def get_funding_exchange():
    import ccxt
    return ccxt.binance({"enableRateLimit": True})
```

- [ ] **Step 4: Verify pass** + full suite green.
- [ ] **Step 5: Commit** `git add bot/funding.py tests/test_funding.py && git commit -m "feat(bot): funding-rate contrarian bias + history helpers"`

---

### Task 2: bot/meanreversion.py funding bias in evaluate

**Files:** Modify `bot/meanreversion.py`, Test `tests/test_meanreversion.py`

- [ ] **Step 1: Failing test**

```python
def test_evaluate_meanrev_entry_requires_funding_alignment(monkeypatch):
    import bot.meanreversion as mr
    # RSI oversold (would enter_long), but funding bias = "short" (crowded longs)
    monkeypatch.setattr(mr, "latest_rsi", lambda df: pd.Series([40.0, 40.0, 24.0]))
    monkeypatch.setattr(mr, "htf_trend_up", lambda df_htf: True)
    monkeypatch.setattr(mr, "htf_trend_down", lambda df_htf: False)
    sig = evaluate_meanrev(None, None, 25, 75, "flat", funding_bias="short")
    assert sig.action == "none"   # blocked: funding says short, RSI says buy -> conflict


def test_evaluate_meanrev_entry_allows_funding_aligned(monkeypatch):
    import bot.meanreversion as mr
    monkeypatch.setattr(mr, "latest_rsi", lambda df: pd.Series([40.0, 40.0, 24.0]))
    monkeypatch.setattr(mr, "htf_trend_up", lambda df_htf: True)
    monkeypatch.setattr(mr, "htf_trend_down", lambda df_htf: False)
    sig = evaluate_meanrev(None, None, 25, 75, "flat", funding_bias="long")
    assert sig.action == "enter_long"
```

- [ ] **Step 2: Verify fail**
- [ ] **Step 3: Implementation** — change `evaluate_meanrev` signature to `evaluate_meanrev(df_htf, df_ltf, oversold, overbought, side="flat", exit_long_rsi=55.0, exit_short_rsi=45.0, funding_bias="neutral")`. In the flat entry branches, add `and (funding_bias == "neutral" or funding_bias == "long")` for long entry and `and (funding_bias == "neutral" or funding_bias == "short")` for short entry. Exits unchanged.
- [ ] **Step 4: Verify pass** + full suite
- [ ] **Step 5: Commit** `git add bot/meanreversion.py tests/test_meanreversion.py && git commit -m "feat(bot): funding bias gate on mean-reversion entries"`

---

### Task 3: funding backtest + calibration

**Files:** Modify `bot/backtest_calibration.py` (mean-reversion path gains funding filter)

- [ ] **Step 1: Extend `run_backtest_meanrev`** with `funding_threshold` param (None = no filter). When set, fetch funding history (Binance first, fallback OKX), align to LTF bars via `reindex(method="ffill")`, compute bias per bar, and require `funding_bias == "long"` (or neutral) for long entries / `"short"` (or neutral) for short entries. Add `--funding` flag to the meanreversion calibration that grid-searches funding_threshold over `[0.0001, 0.0005, 0.001]` plus None. If funding history is too short (< 200 bars), print a warning and use whatever is available.
- [ ] **Step 2: Run** `.venv/bin/python bot/backtest_calibration.py --signal meanreversion --funding`
- [ ] **Step 3: FAIL-LOUD gate** — report trades>=100 AND return>0 both symbols, or the honest negative. Include: funding history length fetched, best config, return/trades per symbol, and compare to the no-funding v3 baseline.
- [ ] **Step 4: Commit** `git add bot/backtest_calibration.py bot/calibration_results_funding.json && git commit -m "feat(bot): funding-filter calibration"`

---

### Task 4: config + loop wiring + docs

**Files:** Modify `bot/config.py`, `bot/loop.py`, `README.md`

- [ ] **Step 1: config** — add `funding_threshold: float = 0.0` (0 = filter disabled). Test default.
- [ ] **Step 2: loop** — in the meanreversion branch, when `cfg.funding_threshold > 0`, fetch current funding rate (via `bot.funding.get_funding_exchange()` + `fetch_funding_rate`) per tick and pass `funding_bias` to `evaluate_meanrev`. Keep momentum path unchanged.
- [ ] **Step 3: README** — append the v4 funding-filter calibration outcome (honest, like v3).
- [ ] **Step 4: Full suite** green + commit.

---

## Self-Review
- Spec coverage: funding.py (T1) ✅; bias gate in evaluate (T2) ✅; calibration with funding (T3) ✅; wiring+docs (T4) ✅.
- Consistency: `funding_bias(rate, threshold)` returns "long"/"short"/"neutral" consistent T1/T2/T3. `evaluate_meanrev(..., funding_bias="neutral")` default keeps existing behavior (backward compat). `funding_threshold=0.0` default = disabled (backward compat). ✅
- Note: Binance `fetch_funding_rate_history` may not exist on all versions; fallback to OKX (95 days available) documented in T3.