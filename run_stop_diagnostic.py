"""Diagnose: is the drawdown stop the trade driver?

Sweep max_drawdown_stop x position size, with trend filter on/off.
"""

from data_cache import load_or_fetch
from feature_engine.builder import build_feature_matrix
from backtest_engine.engine import BacktestEngine
from backtest_engine.rule_strategy import MomentumRuleStrategy

INITIAL = 10000.0

df = load_or_fetch("BTC/USDT", "1d", limit=1000)
features, labels = build_feature_matrix(df, n_bars=5)
years = len(features) / 365.0

# Buy & hold max drawdown for reference
close = features["close"]
roll_peak = close.cummax()
bh_dd = ((close / roll_peak) - 1).min() * 100
print(f"Period: {features.index[0].date()} -> {features.index[-1].date()} ({years:.2f} yrs)")
print(f"Buy&Hold MaxDD: {bh_dd:.1f}%\n")

print(f"{'DDStop':>6} {'Pos%':>5} {'Filter':>7} {'Return%':>9} {'Trades':>7} {'StopExits':>9} {'MaxDD%':>7} {'Sharpe':>7}")
print("-" * 70)

for dd_stop in [30, 50, 100]:
    for pct in [25, 50, 95]:
        for tf in [False, True]:
            engine = BacktestEngine(
                strategy=MomentumRuleStrategy(0.08, -0.07),
                initial_capital=INITIAL,
                timeframe="1d",
                max_position_pct=pct,
                max_drawdown_stop=dd_stop,
                trend_filter=tf,
                min_holding_bars=5,
                cooldown_bars=3,
            )
            r = engine.run(features)
            stops = len([t for t in r.trades if t.get("reason") == "drawdown_stop"])
            print(f"{dd_stop:>6} {pct:>4}% {str(tf):>7} {r.total_return_pct:>8.2f}% {r.total_trades:>7} "
                  f"{stops:>9} {r.max_drawdown_pct:>6.2f}% {r.sharpe_ratio:>7.2f}")
