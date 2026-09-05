"""Phase 1: 1d backtest comparison — baseline vs structural filters.

With the equity bug fixed, re-run everything on 1d timeframe:
1. Buy & hold benchmark
2. Baseline momentum strategy (no filters)
3. Filtered: trend_filter + min_holding + cooldown
"""

import sys
from data_cache import load_or_fetch
from feature_engine.builder import build_feature_matrix
from backtest_engine.engine import BacktestEngine
from backtest_engine.rule_strategy import MomentumRuleStrategy

symbol = "BTC/USDT"
INITIAL = 10000.0

df = load_or_fetch(symbol, "1d", limit=1000)
print(f"Data: {len(df)} candles, {df.index[0].date()} -> {df.index[-1].date()}")

features, labels = build_feature_matrix(df, n_bars=5)
print(f"Usable rows after warmup: {len(features)} ({features.index[0].date()} -> {features.index[-1].date()})")

n_days = len(features)
years = n_days / 365.0

# Benchmark: buy & hold over the same period
bh_start = features["close"].iloc[0]
bh_end = features["close"].iloc[-1]
bh_total = (bh_end / bh_start - 1) * 100
bh_annual = ((bh_end / bh_start) ** (1 / years) - 1) * 100
print(f"\n=== Buy & Hold ({years:.2f} yrs) ===")
print(f"Total: {bh_total:+.2f}%  Annualized: {bh_annual:+.2f}%")

configs = [
    ("Baseline (no filter)", dict()),
    ("Trend filter only", dict(trend_filter=True)),
    ("Trend + min_hold 5 + cooldown 3",
     dict(trend_filter=True, min_holding_bars=5, cooldown_bars=3)),
    ("Trend + min_hold 10 + cooldown 5",
     dict(trend_filter=True, min_holding_bars=10, cooldown_bars=5)),
]

print(f"\n{'Config':<35} {'Return%':>9} {'Annual%':>8} {'Trades':>7} {'MaxDD%':>7} {'Sharpe':>7} {'Win%':>6}")
print("-" * 85)

results = {}
for name, engine_kwargs in configs:
    engine = BacktestEngine(
        strategy=MomentumRuleStrategy(buy_threshold=0.08, sell_threshold=-0.07),
        initial_capital=INITIAL,
        timeframe="1d",
        max_position_pct=95,
        **engine_kwargs,
    )
    r = engine.run(features)
    annual = ((r.final_equity / INITIAL) ** (1 / years) - 1) * 100 if r.final_equity > 0 else -100
    results[name] = (r, annual)
    print(f"{name:<35} {r.total_return_pct:>8.2f}% {annual:>7.2f}% {r.total_trades:>7} "
          f"{r.max_drawdown_pct:>6.2f}% {r.sharpe_ratio:>7.2f} {r.win_rate:>5.1f}%")

# Trade details for the best filtered config
best_name = "Trend + min_hold 5 + cooldown 3"
r = results[best_name][0]
dd_stops = len([t for t in r.trades if t.get("reason") == "drawdown_stop"])
print(f"\n[{best_name}] drawdown-stop exits: {dd_stops}")
print(f"  vs Buy&Hold: {bh_total:+.2f}% — strategy: {r.total_return_pct:+.2f}%")
