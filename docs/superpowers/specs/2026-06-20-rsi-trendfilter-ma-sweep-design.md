# RSI Trend Filter + MA Cross Sweep Design

## Goal
Two independent features:
1. **RSI Trend Filter** — new strategy that only fires RSI signals in trend direction (price above/below MA)
2. **MA Cross parameter sweep** — sweep MACross fast/slow periods

## Part 1: RSITrendFilter Strategy

### New class in `trading/strategy.py`

```python
class RSITrendFilter(Strategy):
    name = "rsi_trend_filter"

    def __init__(self, period=14, overbought=70, oversold=30,
                 trend_period=50, trend_type="sma"):
        self.rsi = RSIThreshold(period, overbought, oversold)
        self.trend_period = trend_period
        self.trend_type = trend_type

    def evaluate(self, overlay, subplots) -> Signal:
        sig = self.rsi.evaluate(overlay, subplots)
        if sig.direction == "中立":
            return sig
        # compute trend MA from close
        close = overlay["close"]
        if self.trend_type == "ema":
            ma = close.ewm(span=self.trend_period).mean()
        else:
            ma = close.rolling(window=self.trend_period).mean()
        current_close = close.iloc[-1]
        current_ma = ma.iloc[-1]
        # only fire in trend direction
        if sig.direction == "偏多" and current_close < current_ma:
            return Signal("中立", 0.0, self.name)
        if sig.direction == "偏空" and current_close > current_ma:
            return Signal("中立", 0.0, self.name)
        return sig
```

### Registration in CustomComposite

Add `"rsi_trend": RSITrendFilter` to `CustomComposite.strategy_map`.

### Config usage

```python
"rsi_trend": {
    "enabled": True,
    "params": {"period": 14, "overbought": 70, "oversold": 30,
               "trend_period": 50, "trend_type": "sma"},
    "weight": 1,
}
```

## Part 2: MA Cross Parameter Sweep

### optimizer.py additions

Same pattern as RSI sweep:

```python
@dataclass
class MaSweepConfig:
    symbol: str = "BTC/USDT"
    timeframe: str = "4h"
    limit: int = 500
    initial_capital: float = 10000.0
    fee_rate: float = 0.001
    slippage: float = 0.0005
    fast_values: list = field(default_factory=lambda: [5, 8, 10, 12, 15, 20])
    slow_values: list = field(default_factory=lambda: [20, 26, 30, 40, 50, 60])

@dataclass
class MaSweepResult:
    params: dict  # { fast, slow }
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    total_trades: int
    final_equity: float
    cagr: float

def run_ma_sweep(config: MaSweepConfig, progress_callback=None) -> list[MaSweepResult]:
    ...
```

Same `_run_single_simulation()` reuse, only `strategy_configs` differs:
```python
strategy_configs = {
    "ma_cross": {"enabled": True, "params": {"fast": f, "slow": s, "type": "ema"}, "weight": 1},
    "rsi": {"enabled": False, "params": {}, "weight": 0},
    "macd": {"enabled": False, "params": {}, "weight": 0},
    "composite": {"enabled": False, "params": {}, "weight": 0},
}
```

### Streamlit UI

`_optimizer_ui()` gains a third radio option:
- 權重掃描 (existing)
- RSI 參數掃描 (existing)
- **MA 參數掃描** (new)

MA mode: fast/slow range selectors + same progress/results/export flow.

## Files Changed
- `trading/strategy.py` — add RSITrendFilter class, register in strategy_map
- `trading/optimizer.py` — add MaSweepConfig, MaSweepResult, run_ma_sweep
- `app.py` — add MA mode to optimizer tab
- `tests/test_strategy.py` — add RSITrendFilter tests
- `tests/test_optimizer.py` — add MA sweep tests

## Testing
- RSITrendFilter: unit test with known data, verify trend filter blocks/signals correctly
- MA sweep: same pattern as RSI sweep tests (test config, test result to_dict, test invalid combo filter)
- Full suite: pytest 47+ tests
