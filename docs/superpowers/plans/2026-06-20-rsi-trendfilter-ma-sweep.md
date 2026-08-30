# RSI Trend Filter + MA Cross Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) for syntax tracking.

**Goal:** Add a trend-filtered RSI strategy and a MA Cross parameter sweep to the optimization system.

**Architecture:** Two independent features sharing existing optimizer infrastructure. RSITrendFilter wraps RSIThreshold + MA direction check. MA sweep follows the same pattern as the existing RSI sweep.

**Tech Stack:** Python 3.9+, pandas, pytest, Streamlit

---

### Task 1: RSITrendFilter — Write failing tests

**Files:**
- Create: `tests/test_strategy.py`

- [ ] **Step 1: Create test file**

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd
from trading.strategy import RSITrendFilter, Signal

def make_overlay(close_prices, rsi_values):
    n = len(close_prices)
    dates = pd.date_range("2025-01-01", periods=n, freq="4h")
    overlay = pd.DataFrame({
        "open": [100.0] * n,
        "high": [max(100.0, c) for c in close_prices],
        "low": [min(100.0, c) for c in close_prices],
        "close": close_prices,
        "volume": [1000.0] * n,
    }, index=dates)
    subplots = {"rsi": pd.DataFrame({"RSI": rsi_values}, index=dates)}
    return overlay, subplots


def test_rsi_trend_bullish_only_in_uptrend():
    close = [90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101]
    rsi = [40, 38, 35, 32, 28, 26, 30, 35, 40, 45, 50, 55]
    overlay, subplots = make_overlay(close, rsi)
    strat = RSITrendFilter(period=14, overbought=70, oversold=30, trend_period=5, trend_type="sma")
    sig = strat.evaluate(overlay, subplots)
    assert sig.direction == "偏多", f"Expected 偏多, got {sig.direction}"
    assert sig.confidence > 0


def test_rsi_trend_bearish_only_in_downtrend():
    close = [110, 108, 106, 104, 102, 100, 98, 96, 94, 92, 90, 88, 86]
    rsi = [60, 65, 70, 75, 80, 78, 72, 68, 62, 58, 55, 50, 45]
    overlay, subplots = make_overlay(close, rsi)
    strat = RSITrendFilter(period=14, overbought=70, oversold=30, trend_period=5, trend_type="sma")
    sig = strat.evaluate(overlay, subplots)
    assert sig.direction == "偏空", f"Expected 偏空, got {sig.direction}"
    assert sig.confidence > 0


def test_rsi_trend_blocks_bullish_in_downtrend():
    close = [101, 100, 99, 98, 97, 96, 95, 94, 93, 92, 91]
    rsi = [40, 38, 35, 32, 28, 26, 30, 35, 40, 45, 50]
    overlay, subplots = make_overlay(close, rsi)
    strat = RSITrendFilter(period=14, overbought=70, oversold=30, trend_period=5, trend_type="sma")
    sig = strat.evaluate(overlay, subplots)
    assert sig.direction == "中立", f"Expected 中立, got {sig.direction}"


def test_rsi_trend_blocks_bearish_in_uptrend():
    close = [90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112]
    rsi = [60, 65, 70, 75, 80, 78, 72, 68, 62, 58, 55, 50]
    overlay, subplots = make_overlay(close, rsi)
    strat = RSITrendFilter(period=14, overbought=70, oversold=30, trend_period=5, trend_type="sma")
    sig = strat.evaluate(overlay, subplots)
    assert sig.direction == "中立", f"Expected 中立, got {sig.direction}"


def test_rsi_trend_uses_ema_when_specified():
    close = [90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101]
    rsi = [40, 38, 35, 32, 28, 26, 30, 35, 40, 45, 50, 55]
    overlay, subplots = make_overlay(close, rsi)
    strat = RSITrendFilter(period=14, overbought=70, oversold=30, trend_period=5, trend_type="ema")
    sig = strat.evaluate(overlay, subplots)
    assert sig.direction == "偏多"
```

Run: `python3 -m pytest tests/test_strategy.py -v`
Expected: 5 FAILED (RSITrendFilter not defined yet)

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/test_strategy.py -v 2>&1 | head -20
```

- [ ] **Step 3: Commit test file**

```bash
git add tests/test_strategy.py && git commit -m "test: add RSITrendFilter tests (failing)"
```


### Task 2: RSITrendFilter — Implement strategy

**Files:**
- Modify: `trading/strategy.py:107` (insert before CustomComposite)

- [ ] **Step 1: Add RSITrendFilter class before CustomComposite**

Insert after line 105 (after CompositeStrategy class):

```python
class RSITrendFilter(Strategy):
    name = "rsi_trend_filter"

    def __init__(self, period: int = 14, overbought: int = 70, oversold: int = 30,
                 trend_period: int = 50, trend_type: str = "sma"):
        self.rsi = RSIThreshold(period, overbought, oversold)
        self.trend_period = trend_period
        self.trend_type = trend_type

    def evaluate(self, overlay: pd.DataFrame, subplots: dict) -> Signal:
        sig = self.rsi.evaluate(overlay, subplots)
        if sig.direction == "中立":
            return sig
        close = overlay["close"]
        if self.trend_type == "ema":
            ma = close.ewm(span=self.trend_period).mean()
        else:
            ma = close.rolling(window=self.trend_period).mean()
        current_close = close.iloc[-1]
        current_ma = ma.iloc[-1]
        if pd.isna(current_ma):
            return Signal("中立", 0.0, self.name)
        if sig.direction == "偏多" and current_close < current_ma:
            return Signal("中立", 0.0, self.name)
        if sig.direction == "偏空" and current_close > current_ma:
            return Signal("中立", 0.0, self.name)
        return sig
```

- [ ] **Step 2: Register in CustomComposite.strategy_map**

Edit line 114 to add `"rsi_trend": RSITrendFilter`:

```python
        strategy_map = {
            "ma_cross": MACross,
            "rsi": RSIThreshold,
            "rsi_trend": RSITrendFilter,
            "macd": MACDCross,
            "composite": CompositeStrategy,
        }
```

- [ ] **Step 3: Run tests**

```bash
python3 -m pytest tests/test_strategy.py -v
```
Expected: 5 PASSED

- [ ] **Step 4: Run full test suite**

```bash
python3 -m pytest tests/ -v --ignore=tests/test_monitor_notifier.py
```
Expected: 48+ passed

- [ ] **Step 5: Commit**

```bash
git add trading/strategy.py && git commit -m "feat: add RSITrendFilter strategy with MA trend filter"
```


### Task 3: MA Sweep — Add engine to optimizer.py

**Files:**
- Modify: `trading/optimizer.py:79` (after RsiSweepResult)

- [ ] **Step 1: Add MaSweepConfig and MaSweepResult dataclasses**

After line 78 (`return asdict(self)` of RsiSweepResult):

```python
@dataclass
class MaSweepConfig:
    symbol: str = "BTC/USDT"
    timeframe: str = "4h"
    limit: int = 500
    initial_capital: float = 10000.0
    fee_rate: float = 0.001
    slippage: float = 0.0005
    risk_config: dict = field(default_factory=lambda: DEFAULT_RISK.copy())
    fast_values: list = field(default_factory=lambda: [5, 8, 10, 12, 15, 20])
    slow_values: list = field(default_factory=lambda: [20, 26, 30, 40, 50, 60])


@dataclass
class MaSweepResult:
    params: dict
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    total_trades: int
    final_equity: float
    cagr: float

    def to_dict(self):
        return asdict(self)
```

- [ ] **Step 2: Add run_ma_sweep function**

Insert after `run_rsi_sweep()` (after line 273):

```python
def run_ma_sweep(
    config: MaSweepConfig,
    progress_callback: Optional[Callable[[int, int, Optional[MaSweepResult]], None]] = None,
) -> list[MaSweepResult]:
    df = fetch_ohlcv(config.symbol, timeframe=config.timeframe, limit=config.limit)
    if df.empty:
        raise ValueError(f"No data for {config.symbol} {config.timeframe}")
    result = compute_all(df)
    overlay = result["overlay"]
    subplots = result["subplots"]

    combos = []
    for fast in config.fast_values:
        for slow in config.slow_values:
            if fast >= slow:
                continue
            combos.append((fast, slow))

    total = len(combos)
    results = []
    count = 0

    for fast, slow in combos:
        strategy_configs = {
            "ma_cross": {"enabled": True, "params": {"fast": fast, "slow": slow, "type": "ema"}, "weight": 1},
            "rsi": {"enabled": False, "params": {}, "weight": 0},
            "macd": {"enabled": False, "params": {}, "weight": 0},
            "composite": {"enabled": False, "params": {}, "weight": 0},
        }
        sr = _run_single_simulation(overlay, subplots, strategy_configs, 0.5, config)
        sr.params = {"fast": fast, "slow": slow}
        results.append(sr)
        count += 1
        if progress_callback:
            progress_callback(count, total, sr)

    results.sort(key=lambda r: r.total_return_pct, reverse=True)
    return results
```

- [ ] **Step 3: Verify import works**

```bash
python3 -c "from trading.optimizer import MaSweepConfig, MaSweepResult, run_ma_sweep; print('OK')"
```
Expected: OK

- [ ] **Step 4: Run full test suite**

```bash
python3 -m pytest tests/ -v --ignore=tests/test_monitor_notifier.py
```
Expected: 47 passed (no MA sweep tests yet, existing tests unaffected)

- [ ] **Step 5: Commit**

```bash
git add trading/optimizer.py && git commit -m "feat: add MA Cross parameter sweep engine"
```


### Task 4: MA Sweep — Write tests

**Files:**
- Modify: `tests/test_optimizer.py`

- [ ] **Step 1: Add imports and tests**

Append to the end of `tests/test_optimizer.py`:

```python
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
    from trading.backtest import _calc_sharpe, _calc_max_drawdown
    from trading.optimizer import _run_single_simulation, MaSweepConfig
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
```

Also add `MaSweepConfig, MaSweepResult, MaSweepResult` to the import line at top of file.

- [ ] **Step 2: Run tests**

```bash
python3 -m pytest tests/test_optimizer.py::test_ma_sweep_result_to_dict tests/test_optimizer.py::test_ma_sweep_config_defaults tests/test_optimizer.py::test_ma_sweep_invalid_combos_skipped -v
```
Expected: 3 PASSED

- [ ] **Step 3: Run full suite**

```bash
python3 -m pytest tests/ -v --ignore=tests/test_monitor_notifier.py
```
Expected: 50 passed

- [ ] **Step 4: Commit**

```bash
git add tests/test_optimizer.py && git commit -m "test: add MA sweep tests"
```


### Task 5: MA Sweep — Streamlit UI

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Read current app.py to see _optimizer_ui**

First confirm the current state:
```bash
grep -n "def _optimizer_ui\|def _weight_sweep\|def _rsi_sweep\|def _display_" app.py
```

- [ ] **Step 2: Modify _optimizer_ui to add MA mode**

In `_optimizer_ui()`, change the radio to:
```python
def _optimizer_ui():
    st.title("⚡ 參數優化")
    opt_mode = st.radio("優化模式", ["權重掃描", "RSI 參數掃描", "MA 參數掃描"], horizontal=True, label_visibility="collapsed")
    if opt_mode == "權重掃描":
        _weight_sweep_ui()
    elif opt_mode == "RSI 參數掃描":
        _rsi_sweep_ui()
    else:
        _ma_sweep_ui()
```

- [ ] **Step 3: Add `_ma_sweep_ui()` function**

```python
def _ma_sweep_ui():
    st.markdown("---")
    from trading.optimizer import MaSweepConfig, MaSweepResult, run_ma_sweep, save_sweep_results

    with st.sidebar:
        st.header("MA 參數設定")
        symbol = st.selectbox("交易對", ["BTC/USDT", "SOL/USDT"], key="ma_symbol")
        tf = st.selectbox("時間框架", ["1h", "4h", "1d"], index=1, key="ma_tf")
        lookback = st.slider("回測K線數", 200, 2000, 500, key="ma_lookback")
        capital = st.number_input("初始資金 ($)", 1000, 100000, 10000, step=1000, key="ma_capital")

        st.subheader("快線 (Fast MA)")
        f_min, f_max = st.select_slider("Fast Period", options=[5, 8, 10, 12, 15, 20],
                                         value=(5, 20), key="ma_fast")

        st.subheader("慢線 (Slow MA > Fast)")
        s_min, s_max = st.select_slider("Slow Period", options=[20, 26, 30, 40, 50, 60],
                                         value=(20, 60), key="ma_slow")

        start_btn = st.button("🚀 開始 MA 優化", type="primary", use_container_width=True)

    if start_btn:
        fast_list = [v for v in [5, 8, 10, 12, 15, 20] if f_min <= v <= f_max]
        slow_list = [v for v in [20, 26, 30, 40, 50, 60] if s_min <= v <= s_max]

        config = MaSweepConfig(
            symbol=symbol, timeframe=tf, limit=lookback,
            initial_capital=float(capital),
            fast_values=fast_list, slow_values=slow_list,
        )

        total_combos = sum(1 for f in fast_list for s in slow_list if f < s)
        st.info(f"測試組合共 **{total_combos:,}** 組")

        progress_bar = st.progress(0, text="初始化...")
        best_text = st.empty()
        all_results = []

        def on_progress(cur, total, latest):
            progress_bar.progress(cur / total, text=f"{cur}/{total}")
            if latest:
                all_results.append(latest)
                best = max(all_results, key=lambda r: r.total_return_pct)
                best_text.text(f"當前最佳報酬: {best.total_return_pct:+.2f}%  "
                               f"(Sharpe: {best.sharpe_ratio:.2f}  DD: {best.max_drawdown_pct:.1f}%)")

        try:
            results = run_ma_sweep(config, on_progress)
        except Exception as e:
            st.error(f"優化失敗: {e}")
            return

        progress_bar.empty()
        st.success(f"✅ 完成！共測試 {len(results):,} 組")
        save_path = save_sweep_results(results, config)
        st.caption(f"結果已儲存: {save_path}")
        _display_ma_results(results, config)
```

- [ ] **Step 4: Add `_display_ma_results()`**

```python
def _display_ma_results(results, config):
    st.subheader("🏆 Top 30 最佳參數")
    top = results[:30]
    rows = []
    for i, r in enumerate(top):
        p = r.params
        rows.append({
            "排名": i + 1,
            "Fast": p.get("fast", "-"),
            "Slow": p.get("slow", "-"),
            "報酬率": f"{r.total_return_pct:+.2f}%",
            "Sharpe": f"{r.sharpe_ratio:.2f}",
            "MaxDD": f"{r.max_drawdown_pct:.2f}%",
            "勝率": f"{r.win_rate:.1f}%",
            "交易": r.total_trades,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("最佳參數詳情")
    st.json(results[0].to_dict())

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ 套用最佳參數", type="primary", use_container_width=True):
            _apply_ma_params(results[0])
    with col2:
        st.download_button(
            "📥 下載完整結果 CSV",
            data=pd.DataFrame([r.to_dict() for r in results]).to_csv(index=False).encode(),
            file_name=f"ma_sweep_{config.symbol.replace('/','_')}_{config.timeframe}.csv",
            mime="text/csv", use_container_width=True,
        )


def _apply_ma_params(best):
    import yaml
    from pathlib import Path
    config_path = Path(__file__).parent / "monitor" / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    p = best.params
    strategies = cfg.setdefault("trading", {}).setdefault("strategies", {})
    ma_cfg = strategies.setdefault("ma_cross", {})
    ma_cfg["params"] = {"fast": p["fast"], "slow": p["slow"], "type": "ema"}
    ma_cfg["enabled"] = True
    ma_cfg["weight"] = 1
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    st.success("✅ MA 參數已套用至 monitor/config.yaml")
```

- [ ] **Step 5: Verify syntax**

```bash
python3 -c "import ast; ast.parse(open('app.py').read()); print('Syntax OK')"
```

- [ ] **Step 6: Run full test suite**

```bash
python3 -m pytest tests/ -v --ignore=tests/test_monitor_notifier.py
```
Expected: 50 passed

- [ ] **Step 7: Commit**

```bash
git add app.py && git commit -m "feat: add MA Cross parameter sweep UI"
```


### Task 6: Push + run MA sweep + Notion report

**Files:**
- N/A (deployment + data collection)

- [ ] **Step 1: Push to GitHub**

```bash
git push
```

- [ ] **Step 2: Run MA sweep live**

```bash
python3 << 'PYEOF'
import sys; sys.path.insert(0, ".")
from trading.optimizer import MaSweepConfig, run_ma_sweep, save_sweep_results

config = MaSweepConfig(
    symbol="BTC/USDT", timeframe="4h", limit=500, initial_capital=10000.0,
    fast_values=[5, 8, 10, 12, 15, 20],
    slow_values=[20, 26, 30, 40, 50, 60],
)

def on_progress(cur, total, latest):
    bar = "█" * int(40 * cur / total) + "░" * (40 - int(40 * cur / total))
    print(f"\r[{bar}] {cur}/{total}", end="", flush=True)

results = run_ma_sweep(config, on_progress)
print(f"\nDone! {len(results)} results")
save_path = save_sweep_results(results, config)
print(f"Saved: {save_path}")

for i, r in enumerate(results[:10]):
    p = r.params
    print(f"{i+1:2}. Fast={p['fast']:2} Slow={p['slow']:2}  "
          f"Ret={r.total_return_pct:+.2f}% Sharpe={r.sharpe_ratio:.2f}  "
          f"DD={r.max_drawdown_pct:.2f}% WR={r.win_rate:.1f}% Trades={r.total_trades}")
PYEOF
```

- [ ] **Step 3: Create Notion report**

Search for the parent page, then create a report page with results (following existing pattern).
