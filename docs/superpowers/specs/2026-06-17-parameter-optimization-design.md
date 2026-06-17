# Parameter Optimization for Auto Trading

## Goal
Build a parameter sweep/optimization system that finds the optimal strategy parameters by testing all combinations of weights and thresholds, ranked by total return %.

## Scope (Phase D only)
- CustomComposite's 4 sub-strategies (ma_cross, rsi, macd, composite)
- Weight range: 0–3 per sub-strategy (4 levels → 4^4 = 256 combos)
- Threshold range: 0.3–0.9, step 0.1 (7 levels)
- Total: 256 × 7 = 1792 combinations
- Metric: total return % (primary sort), Sharpe, Max DD, Win Rate tracked

Future phases (C, B, A) will follow the same optimizer pattern.

## Architecture

### New file: `trading/optimizer.py`

Core optimization logic, no UI dependencies.

```python
@dataclass
class ParamSweepConfig:
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    limit: int = 500
    initial_capital: float = 10000.0
    fee_rate: float = 0.001
    risk_config: dict = field(default_factory=lambda: DEFAULT_RISK)
    # strategy weight range
    strategy_order: list = field(default_factory=lambda: ["ma_cross", "rsi", "macd", "composite"])
    weight_range: tuple = (0, 3)      # inclusive
    threshold_range: tuple = (0.3, 0.9, 0.1)  # start, stop, step

@dataclass
class SweepResult:
    params: dict      # { "ma_cross_weight": 2, "rsi_weight": 1, ..., "threshold": 0.6 }
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    total_trades: int
    final_equity: float
    cagr: float

def run_parameter_sweep(config: ParamSweepConfig) -> list[SweepResult]:
    """
    1. Fetch OHLCV once
    2. Compute indicators once → overlay, subplots
    3. For each weight x threshold combo:
       - Build strategy config from weights
       - Run portfolio simulation on pre-computed indicators
       - Collect metrics
    4. Sort by total_return_pct descending
    5. Return results
    """
```

### Efficiency
- Data fetch and indicator computation happen ONCE
- Inner loop only re-evaluates strategies and updates portfolio
- Progress callback for Streamlit integration
- ~1792 iterations should complete in 30-60 seconds (mostly portfolio simulation)

### Streamlit UI Changes
File: `app.py` — `backtest_page()` gets a new tab:

```
[單次回測] [⚡ 參數優化]
```

**Optimization tab:**
- Symbol, timeframe, lookback selectors (same as single backtest)
- For each sub-strategy: weight slider 0–3 (step 1), default 0–1–2–3
- Threshold slider: 0.3–0.9 step 0.1, default 0.3–0.7–0.9
- "開始優化" button with progress bar + estimated time
- Results table: Top 30 sorted by total return %
  - Columns: peso weights + threshold + return% + Sharpe + DD + Win Rate + trades
  - Best row highlighted, click to expand detail
- "套用最佳參數" — copies best params to `monitor/config.yaml`
- "匯出到 Notion" — records top results to a Notion page
- "下載完整結果" — exports full 1792-row CSV

### Notion Integration
On "匯出到 Notion":
1. Create a page under a specified parent
2. Page title: `參數優化 report — 2026-06-17`
3. Content includes:
   - Metadata: symbol, timeframe, lookback, total combos tested, date
   - Table of Top 10 results with all metrics
   - Best parameter set highlighted
4. Uses the existing Notion MCP tools

### Persistence
- Results auto-saved to `trading/optimization_results/<date>_<symbol>_<tf>.json`
- Full results + summary can be reloaded in a future session

### Error Handling
- If data fetch fails: abort with clear error message
- If a specific parameter combo produces NaN/inf metrics: skip entry with warning count
- Progress bar updates every ~50 iterations

## Files Changed
- `trading/optimizer.py` (new)
- `app.py` — backtest_page() gets optimizer tab
- `monitor/config.yaml` — can be updated by "apply best" button

## Not in Scope (First Iteration)
- C/B/A phase strategy-level parameter optimization (will follow same pattern)
- Live Notion database creation (uses page-level record, not full DB)
- Multi-symbol simultaneous optimization
- Advanced sampling (Bayesian/evolutionary)
