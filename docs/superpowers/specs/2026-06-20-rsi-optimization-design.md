# RSI Parameter Optimization Design

## Goal
Phase B of strategy optimization: sweep RSIThreshold parameters (period, overbought, oversold) to find the combination that maximizes total return %.

## Scope
- Strategy: RSIThreshold only (MA/MACD/Composite disabled)
- Parameters: period (7~21, step 2), overbought (65~85, step 5), oversold (15~35, step 5)
- Total: ~200 valid combos (filtering oversold >= overbought)
- Metric: total return % (same as Phase D)
- Other settings (risk, fee, slippage) use defaults

## Architecture

### optimizer.py addition
New function `run_rsi_sweep()` following the same fetch-once, compute-once pattern:

```python
@dataclass
class RsiSweepConfig:
    symbol: str = "BTC/USDT"
    timeframe: str = "4h"
    limit: int = 500
    initial_capital: float = 10000.0
    fee_rate: float = 0.001
    slippage: float = 0.0005
    period_values: list = field(default_factory=lambda: [7, 9, 11, 13, 14, 15, 17, 19, 21])
    ob_values: list = field(default_factory=lambda: [65, 70, 75, 80, 85])
    os_values: list = field(default_factory=lambda: [15, 20, 25, 30, 35])

@dataclass
class RsiSweepResult:
    params: dict  # { period, overbought, oversold }
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    total_trades: int
    final_equity: float
    cagr: float

def run_rsi_sweep(config: RsiSweepConfig, progress_callback=None) -> list[RsiSweepResult]:
    # Fetch + compute once
    # For each valid (period, ob, os) combo:
    #   Create RSIThreshold with those params
    #   Run single-strategy simulation (only RSI, no composite)
    #   Collect metrics
    # Sort by total_return_pct desc
```

### Streamlit UI change
The optimizer tab adds a second mode via radio:
- **權重優化** (existing)
- **RSI 參數優化** (new)

RSI mode shows:
- Period range slider (min 7, max 21, step 2)
- Overbought range slider (65~85, step 5)
- Oversold range slider (15~35, step 5)
- Same progress/results/apply/export flow

### Reuse
Maximizes reuse of Phase D infrastructure:
- Same `_run_single_simulation()` for portfolio simulation
- Same `save_sweep_results()` / `load_sweep_results()` for persistence
- Same Streamlit UI pattern (progress bar, Top 30 table, CSV export, Notion)
- `_apply_best_params()` writes best RSI params to config.yaml

## Files Changed
- `trading/optimizer.py` — add RsiSweepConfig, RsiSweepResult, run_rsi_sweep
- `app.py` — add RSI mode to optimizer tab
- `tests/test_optimizer.py` — add RSI sweep tests

## Next Steps
Phase B → Phase A (MA Cross parameter optimization)
