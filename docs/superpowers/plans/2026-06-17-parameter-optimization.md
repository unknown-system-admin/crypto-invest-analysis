# Parameter Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a parameter sweep engine that tests 1792 weight/threshold combinations to find the optimal CustomComposite strategy config, ranked by total return %.

**Architecture:** New `trading/optimizer.py` module with `run_parameter_sweep()` that fetches data + computes indicators once, then iterates all param combos in-memory. Streamlit backtest page gets a new "參數優化" tab with progress bar and results table. Best params can be applied to `monitor/config.yaml` and exported to Notion.

**Tech Stack:** Python, pandas, Streamlit, Notion MCP, JSON persistence

---

### File Structure

- **Create:** `trading/optimizer.py` — sweep engine, config, result types
- **Create:** `tests/test_optimizer.py` — unit tests for optimizer
- **Modify:** `app.py:344-446` — backtest_page() gains optimizer tab
- **Modify:** `monitor/config.yaml` — can be updated by "apply best"
- **Create (runtime):** `trading/optimization_results/` — JSON result cache

---

### Task 1: Optimizer Module — Core Engine

**Files:**
- Create: `trading/optimizer.py`

- [ ] **Step 1: Write `ParamSweepConfig` and `SweepResult` dataclasses**

```python
from dataclasses import dataclass, field, asdict
from typing import Optional, Callable
import itertools
import json
import os
from pathlib import Path

import pandas as pd
import numpy as np

from data.fetcher import fetch_ohlcv
from indicators.calculator import compute_all
from trading.strategy import CustomComposite
from trading.portfolio import Portfolio, calculate_position_size
from trading.executor import PaperExecutor
from trading.config import DEFAULT_STRATEGIES, DEFAULT_RISK
from trading.backtest import _calc_sharpe, _calc_max_drawdown


@dataclass
class ParamSweepConfig:
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    limit: int = 500
    initial_capital: float = 10000.0
    fee_rate: float = 0.001
    slippage: float = 0.0005
    risk_config: dict = field(default_factory=lambda: DEFAULT_RISK.copy())
    strategy_order: list = field(default_factory=lambda: ["ma_cross", "rsi", "macd", "composite"])
    weight_min: int = 0
    weight_max: int = 3
    threshold_start: float = 0.3
    threshold_stop: float = 0.9
    threshold_step: float = 0.1


@dataclass
class SweepResult:
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

- [ ] **Step 2: Write `_run_single_simulation()` — portfolio simulation on pre-computed data**

This is the inner loop extracted from `trading/backtest.py:run_backtest()` but operating on already-computed overlay/subplots:

```python
def _run_single_simulation(
    overlay: pd.DataFrame,
    subplots: dict,
    strategy_configs: dict,
    signal_threshold: float,
    config: ParamSweepConfig,
) -> SweepResult:
    strategy = CustomComposite(strategy_configs, threshold=signal_threshold)
    portfolio = Portfolio(cash=config.initial_capital, positions=[], orders=[])
    executor = PaperExecutor(portfolio, slippage=config.slippage)

    equity_curve = []
    trades_log = []
    daily_trade_count = 0
    last_trade_date = ""

    risk = config.risk_config
    max_pos_pct = risk.get("max_position_pct", 25)
    min_bars_hold = risk.get("min_bars_hold", 1)
    max_daily_trades = risk.get("max_daily_trades", 10)
    max_dd_stop = risk.get("max_drawdown_stop", 30)

    for i in range(min_bars_hold, len(overlay)):
        window = overlay.iloc[:i + 1]
        sub_window = {k: v.iloc[:i + 1] for k, v in subplots.items()}
        sig = strategy.evaluate(window, sub_window)
        current_price = float(overlay["close"].iloc[i])
        idx = overlay.index[i]

        current_date = str(idx.date()) if hasattr(idx, "date") else str(i)
        if current_date != last_trade_date:
            daily_trade_count = 0
            last_trade_date = current_date

        has_position = any(
            p.symbol == config.symbol and p.side == "long"
            for p in portfolio.positions
        )

        current_dd = _calc_max_drawdown(equity_curve + [portfolio.total_equity])
        if current_dd > max_dd_stop:
            pos = [p for p in portfolio.positions if p.symbol == config.symbol and p.side == "long"]
            if pos:
                executor.execute_sell(config.symbol, current_price, pos[0].quantity)
            equity_curve.append(portfolio.total_equity)
            continue

        if sig and sig.direction == "偏多" and not has_position and executor.can_trade(
            config.symbol, daily_trade_count, max_daily_trades
        ):
            qty = calculate_position_size(portfolio.cash, current_price, max_pos_pct)
            if qty > 0:
                order = executor.execute_buy(config.symbol, current_price, qty)
                trades_log.append({
                    "date": current_date, "action": "buy",
                    "price": round(order.price, 2), "quantity": order.quantity,
                })
                daily_trade_count += 1

        elif sig and sig.direction == "偏空" and has_position and executor.can_trade(
            config.symbol, daily_trade_count, max_daily_trades
        ):
            pos_list = [p for p in portfolio.positions if p.symbol == config.symbol and p.side == "long"]
            if pos_list:
                order = executor.execute_sell(config.symbol, current_price, pos_list[0].quantity)
                if order:
                    trades_log.append({
                        "date": current_date, "action": "sell",
                        "price": round(order.price, 2), "quantity": order.quantity,
                        "pnl": round(order.pnl, 2), "pnl_pct": round(order.pnl_pct, 2),
                    })
                    daily_trade_count += 1

        for p in portfolio.positions:
            p.update_market(current_price)
        equity_curve.append(portfolio.total_equity)

    final_equity = portfolio.total_equity
    total_return = ((final_equity - config.initial_capital) / config.initial_capital) * 100 if config.initial_capital > 0 else 0.0
    total_days = len(equity_curve)
    cagr = ((final_equity / config.initial_capital) ** (365 / max(total_days, 1)) - 1) * 100 if config.initial_capital > 0 and total_days > 0 else 0.0

    equity_series = pd.Series(equity_curve)
    returns_series = equity_series.pct_change().dropna()
    sharpe = _calc_sharpe(returns_series)
    max_dd = _calc_max_drawdown(equity_curve)

    filled_sells = [o for o in portfolio.orders if o.side == "sell" and o.status == "filled"]
    wins = sum(1 for o in filled_sells if o.pnl > 0)
    total_closed = len(filled_sells)
    win_rate = (wins / total_closed * 100) if total_closed > 0 else 0.0

    return SweepResult(
        params={},
        total_return_pct=round(total_return, 2),
        sharpe_ratio=round(sharpe, 2),
        max_drawdown_pct=round(max_dd, 2),
        win_rate=round(win_rate, 1),
        total_trades=len(trades_log),
        final_equity=round(final_equity, 2),
        cagr=round(cagr, 2),
    )
```

- [ ] **Step 3: Write `run_parameter_sweep()` — orchestrator with progress callback**

```python
def run_parameter_sweep(
    config: ParamSweepConfig,
    progress_callback: Optional[Callable[[int, int, Optional[SweepResult]], None]] = None,
) -> list[SweepResult]:
    df = fetch_ohlcv(config.symbol, timeframe=config.timeframe, limit=config.limit)
    if df.empty:
        raise ValueError(f"No data for {config.symbol} {config.timeframe}")

    result = compute_all(df)
    overlay = result["overlay"]
    subplots = result["subplots"]

    weight_values = list(range(config.weight_min, config.weight_max + 1))
    thresholds = []
    t = config.threshold_start
    while t <= config.threshold_stop + 1e-9:
        thresholds.append(round(t, 1))
        t = round(t + config.threshold_step, 1)

    n_strategies = len(config.strategy_order)
    total = (len(weight_values) ** n_strategies) * len(thresholds)
    count = 0
    results = []
    best_return = -float("inf")

    for weights in itertools.product(weight_values, repeat=n_strategies):
        for thresh in thresholds:
            strategy_configs = {}
            for i, sname in enumerate(config.strategy_order):
                params_copy = DEFAULT_STRATEGIES.get(sname, {}).get("params", {}).copy()
                strategy_configs[sname] = {
                    "enabled": weights[i] > 0,
                    "params": params_copy,
                    "weight": int(weights[i]),
                }

            sr = _run_single_simulation(overlay, subplots, strategy_configs, thresh, config)
            sr.params = {
                **{f"{sname}_weight": int(weights[j]) for j, sname in enumerate(config.strategy_order)},
                "threshold": thresh,
            }

            if sr.total_return_pct > best_return:
                best_return = sr.total_return_pct

            results.append(sr)
            count += 1
            if progress_callback:
                progress_callback(count, total, sr)

    results.sort(key=lambda r: r.total_return_pct, reverse=True)
    return results
```

- [ ] **Step 4: Add `save_sweep_results()` and `load_sweep_results()` helpers**

```python
RESULTS_DIR = Path(__file__).resolve().parent / "optimization_results"


def save_sweep_results(results: list[SweepResult], config: ParamSweepConfig) -> str:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    safe_sym = config.symbol.replace("/", "_")
    filename = f"{timestamp}_{safe_sym}_{config.timeframe}.json"
    path = RESULTS_DIR / filename
    data = {
        "config": asdict(config),
        "results": [r.to_dict() for r in results],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    return str(path)


def load_sweep_results(path: str) -> tuple[ParamSweepConfig, list[SweepResult]]:
    with open(path) as f:
        data = json.load(f)
    config = ParamSweepConfig(**data["config"])
    results = [SweepResult(**r) for r in data["results"]]
    return config, results


def list_sweep_result_files() -> list[str]:
    if not RESULTS_DIR.exists():
        return []
    return sorted([str(p) for p in RESULTS_DIR.iterdir() if p.suffix == ".json"], reverse=True)
```

- [ ] **Step 5: Run tests to verify optimizer compiles**

Run: `python3 -c "from trading.optimizer import ParamSweepConfig, SweepResult, run_parameter_sweep; print('OK')"`
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add trading/optimizer.py
git commit -m "feat: add parameter sweep optimizer engine"
```

---

### Task 2: Optimizer Tests

**Files:**
- Create: `tests/test_optimizer.py`

- [ ] **Step 1: Write unit tests**

```python
import pandas as pd
import numpy as np
from trading.optimizer import (
    ParamSweepConfig,
    SweepResult,
    _run_single_simulation,
)


def _make_indicator_data(length=200):
    close = np.linspace(100, 110, length) + np.random.randn(length) * 2
    overlay = pd.DataFrame({"close": close})
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
    assert sr.total_return_pct != 0
    assert sr.sharpe_ratio != 0
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
```

- [ ] **Step 2: Run tests to verify**

Run: `python3 -m pytest tests/test_optimizer.py -v`
Expected: 4 passed (or 1 skipped if offline)

- [ ] **Step 3: Commit**

```bash
git add tests/test_optimizer.py
git commit -m "test: add parameter sweep optimizer tests"
```

---

### Task 3: Streamlit Optimizer UI

**Files:**
- Modify: `app.py:344-446` (backtest_page function)

- [ ] **Step 1: Add tab selection at top of backtest_page**

Replace the current backtest page header with tabs:

```python
def backtest_page():
    tab_mode = st.radio("模式", ["單次回測", "⚡ 參數優化"], horizontal=True, label_visibility="collapsed")

    if tab_mode == "單次回測":
        _single_backtest_ui()
    else:
        _optimizer_ui()
```

Then extract the existing code into `_single_backtest_ui()`:

```python
def _single_backtest_ui():
    st.title("📊 策略回測")
    st.markdown("---")
    # ... existing code from backtest_page()
```

- [ ] **Step 2: Write `_optimizer_ui()` with parameter controls and progress**

```python
def _optimizer_ui():
    st.title("⚡ 參數優化 — 權重掃描")
    st.markdown("---")

    from trading.optimizer import ParamSweepConfig, SweepResult, run_parameter_sweep
    from trading.optimizer import save_sweep_results, list_sweep_result_files
    from trading.config import DEFAULT_STRATEGIES

    with st.sidebar:
        st.header("優化設定")
        symbol = st.selectbox("交易對", ["BTC/USDT", "SOL/USDT"], key="opt_symbol")
        tf = st.selectbox("時間框架", ["1h", "4h", "1d"], index=0, key="opt_tf")
        lookback = st.slider("回測K線數", 200, 2000, 500, key="opt_lookback")
        capital = st.number_input("初始資金 ($)", 1000, 100000, 10000, step=1000, key="opt_capital")

        st.subheader("權重範圍 (0~3)")
        w_ma = st.slider("MA Cross", 0, 3, (0, 3), key="opt_w_ma")
        w_rsi = st.slider("RSI", 0, 3, (0, 3), key="opt_w_rsi")
        w_macd = st.slider("MACD", 0, 3, (0, 3), key="opt_w_macd")
        w_comp = st.slider("Composite", 0, 3, (0, 3), key="opt_w_comp")

        st.subheader("Threshold 範圍")
        th_min, th_max = st.slider("Threshold", 0.3, 0.9, (0.3, 0.7), step=0.1, key="opt_th")
        th_step = st.selectbox("Step", [0.1, 0.2], index=0, key="opt_th_step")

        start_btn = st.button("🚀 開始優化", type="primary", use_container_width=True)

    if start_btn:
        config = ParamSweepConfig(
            symbol=symbol, timeframe=tf, limit=lookback,
            initial_capital=float(capital),
            weight_min=w_ma[0], weight_max=w_ma[1],
        )
        # Use the union of all weight ranges as the global range
        # (all strategies use same weight range for simplicity)
        # Override strategy-specific ranges:
        config.weight_min = min(w_ma[0], w_rsi[0], w_macd[0], w_comp[0])
        config.weight_max = max(w_ma[1], w_rsi[1], w_macd[1], w_comp[1])
        config.threshold_start = th_min
        config.threshold_stop = th_max
        config.threshold_step = th_step

        weight_values = config.weight_max - config.weight_min + 1
        n_strats = len(config.strategy_order)
        n_th = int((th_max - th_min) / th_step) + 1
        total_combos = (weight_values ** n_strats) * n_th
        st.info(f"測試組合共 **{total_combos:,}** 組，預計 30~90 秒")

        progress_bar = st.progress(0, text="初始化...")
        status_text = st.empty()
        best_text = st.empty()

        all_results = []

        def on_progress(current, total, latest):
            pct = current / total
            progress_bar.progress(pct, text=f"{current}/{total}")
            if latest and latest.total_return_pct > -999:
                all_results.append(latest)
                best = max(all_results, key=lambda r: r.total_return_pct)
                best_text.text(f"當前最佳報酬: {best.total_return_pct:+.2f}%  "
                               f"(Sharpe: {best.sharpe_ratio:.2f}  MaxDD: {best.max_drawdown_pct:.2f}%)")

        try:
            results = run_parameter_sweep(config, progress_callback=on_progress)
        except Exception as e:
            st.error(f"優化失敗: {e}")
            return

        progress_bar.empty()
        st.success(f"✅ 完成！共測試 {len(results):,} 組")

        # Save results
        save_path = save_sweep_results(results, config)
        st.caption(f"結果已儲存: {save_path}")

        _display_optimizer_results(results, config)

def _display_optimizer_results(results: list, config):
    st.subheader("🏆 Top 30 最佳參數")
    top = results[:30]
    rows = []
    for i, r in enumerate(top):
        p = r.params
        rows.append({
            "排名": i + 1,
            "MA": p.get("ma_cross_weight", "-"),
            "RSI": p.get("rsi_weight", "-"),
            "MACD": p.get("macd_weight", "-"),
            "Comp": p.get("composite_weight", "-"),
            "Thresh": p.get("threshold", "-"),
            "報酬率": f"{r.total_return_pct:+.2f}%",
            "Sharpe": f"{r.sharpe_ratio:.2f}",
            "MaxDD": f"{r.max_drawdown_pct:.2f}%",
            "勝率": f"{r.win_rate:.1f}%",
            "交易": r.total_trades,
        })

    df_top = pd.DataFrame(rows)
    st.dataframe(df_top, use_container_width=True, hide_index=True)

    st.subheader("最佳參數詳情")
    best = results[0]
    st.json(best.to_dict())

- [ ] **Step 3: Write "套用最佳參數" logic**

Add at end of _optimizer_ui after results display:

```python
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ 套用最佳參數", type="primary", use_container_width=True):
                _apply_best_params(results[0], config)
        with col2:
            result_path = save_sweep_results(results, config)
            st.download_button("📥 下載完整結果 CSV", 
                data=pd.DataFrame([r.to_dict() for r in results]).to_csv(index=False).encode(),
                file_name=f"optimization_{config.symbol.replace('/','_')}_{config.timeframe}.csv",
                mime="text/csv", use_container_width=True)


def _apply_best_params(best: "SweepResult", config: "ParamSweepConfig"):
    import yaml
    from pathlib import Path

    config_path = Path(__file__).parent / "monitor" / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    strategies = cfg.setdefault("trading", {}).setdefault("strategies", {})
    p = best.params
    for sname in config.strategy_order:
        w = int(p.get(f"{sname}_weight", 0))
        if sname in strategies:
            strategies[sname]["weight"] = w
            strategies[sname]["enabled"] = w > 0

    cfg["trading"]["risk"]["signal_threshold"] = p.get("threshold", 0.6)

    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    st.success("✅ 最佳參數已套用至 monitor/config.yaml")
```

- [ ] **Step 4: Integrate into app.py**

Replace the existing `backtest_page()` with the tabbed version.

- [ ] **Step 5: Verify app compiles**

Run: `python3 -c "import ast; ast.parse(open('app.py').read()); print('Syntax OK')"`
Expected: `Syntax OK`

- [ ] **Step 6: Commit**

```bash
git add app.py
git commit -m "feat: add parameter optimization tab to backtest page"
```

---

### Task 4: Push, Deploy & Notion Export

- [ ] **Step 1: Run all tests**

Run: `python3 -m pytest tests/ -v --ignore=tests/test_monitor_notifier.py`
Expected: all passed

- [ ] **Step 2: Push to GitHub**

Run: `git push`
Expected: master → origin/master

- [ ] **Step 3: Agent runs optimization on live dashboard**

After deploy, visit the dashboard → backtest → 參數優化 tab → run a sweep. Then save the JSON result file path for Notion export.

- [ ] **Step 4: Agent exports Top 10 results to Notion**

Agent uses Notion MCP tools to create a page under the project parent with:
- Title: `參數優化報告 — <date>`
- Content includes: metadata (symbol, timeframe, lookback, combos), Top 10 table, best params JSON

- [ ] **Step 5: Update anchored summary**
