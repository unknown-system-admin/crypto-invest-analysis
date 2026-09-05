"""Phase 2: cost-aware grid search on 1d.

Objective: net total return (fees + slippage already embedded in engine).
No hard trade-count cap — costs penalize overtrading naturally (user's call).

Grid: buy/sell thresholds x min_holding x cooldown x trend_filter
Fixed: dd_stop=50, position=95%, 1d timeframe.
"""

import time
from itertools import product
from data_cache import load_or_fetch
from feature_engine.builder import build_feature_matrix
from backtest_engine.engine import BacktestEngine
from backtest_engine.rule_strategy import MomentumRuleStrategy

INITIAL = 10000.0

df = load_or_fetch("BTC/USDT", "1d", limit=1000)
features, labels = build_feature_matrix(df, n_bars=5)
years = len(features) / 365.0

bh_ret = (features["close"].iloc[-1] / features["close"].iloc[0] - 1) * 100
print(f"Period: {features.index[0].date()} -> {features.index[-1].date()} ({years:.2f} yrs)")
print(f"Buy&Hold: {bh_ret:+.2f}%\n")

buy_ths = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
sell_ths = [-0.05, -0.10, -0.15, -0.20, -0.25, -0.30]
holds = [0, 5, 10]
cools = [0, 3]
filters = [False, True]

total = len(buy_ths) * len(sell_ths) * len(holds) * len(cools) * len(filters)
print(f"Grid: {total} configs")
print(f"{'buy':>5} {'sell':>6} {'hold':>5} {'cool':>5} {'filt':>5} {'Ret%':>8} {'Trades':>7} {'MaxDD%':>7} {'Sharpe':>7} {'Win%':>6}")
print("-" * 68)

results = []
t0 = time.time()
for buy, sell, hold, cool, tf in product(buy_ths, sell_ths, holds, cools, filters):
    engine = BacktestEngine(
        strategy=MomentumRuleStrategy(buy_threshold=buy, sell_threshold=sell),
        initial_capital=INITIAL,
        timeframe="1d",
        max_position_pct=95,
        max_drawdown_stop=50,
        trend_filter=tf,
        min_holding_bars=hold,
        cooldown_bars=cool,
    )
    r = engine.run(features)
    results.append({
        "buy": buy, "sell": sell, "hold": hold, "cool": cool, "filt": tf,
        "ret": r.total_return_pct, "trades": r.total_trades,
        "maxdd": r.max_drawdown_pct, "sharpe": r.sharpe_ratio,
        "win": r.win_rate, "final": r.final_equity,
    })

elapsed = time.time() - t0
print(f"\nDone in {elapsed:.0f}s ({elapsed/total:.2f}s per run)")

# Sort by net return
results.sort(key=lambda x: x["ret"], reverse=True)

print(f"\n=== TOP 15 (by net return) ===")
print(f"{'buy':>5} {'sell':>6} {'hold':>5} {'cool':>5} {'filt':>5} {'Ret%':>8} {'Trades':>7} {'MaxDD%':>7} {'Sharpe':>7} {'Win%':>6}")
for x in results[:15]:
    print(f"{x['buy']:>5.2f} {x['sell']:>6.2f} {x['hold']:>5} {x['cool']:>5} {str(x['filt']):>5} "
          f"{x['ret']:>7.2f}% {x['trades']:>7} {x['maxdd']:>6.2f}% {x['sharpe']:>7.2f} {x['win']:>5.1f}%")

n_beat_bh = sum(1 for x in results if x["ret"] > bh_ret)
print(f"\nConfigs beating Buy&Hold (+{bh_ret:.2f}%): {n_beat_bh}/{total}")
print(f"Median return: {sorted(x['ret'] for x in results)[total//2]:.2f}%")
