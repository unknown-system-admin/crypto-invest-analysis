"""Walk-forward validation: select params on train, evaluate on unseen test.

Train: first 60% of data. Test: last 40% (completely unseen during selection).
"""

from itertools import product
from data_cache import load_or_fetch
from feature_engine.builder import build_feature_matrix
from backtest_engine.engine import BacktestEngine
from backtest_engine.rule_strategy import MomentumRuleStrategy

INITIAL = 10000.0

df = load_or_fetch("BTC/USDT", "1d", limit=1000)
features, labels = build_feature_matrix(df, n_bars=5)

split = int(len(features) * 0.6)
train = features.iloc[:split]
test = features.iloc[split:]
print(f"Train: {train.index[0].date()} -> {train.index[-1].date()} ({len(train)} bars)")
print(f"Test:  {test.index[0].date()} -> {test.index[-1].date()} ({len(test)} bars)")

# Buy & hold on each period
bh_train = (train["close"].iloc[-1] / train["close"].iloc[0] - 1) * 100
bh_test = (test["close"].iloc[-1] / test["close"].iloc[0] - 1) * 100
print(f"B&H train: {bh_train:+.2f}%  |  B&H test: {bh_test:+.2f}%\n")

buy_ths = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
sell_ths = [-0.05, -0.10, -0.15, -0.20, -0.25, -0.30]
holds = [0, 5, 10]
cools = [0, 3]
filters = [False, True]


def run_one(df_part, buy, sell, hold, cool, tf):
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
    return engine.run(df_part)


# Select best on train
best = None
for buy, sell, hold, cool, tf in product(buy_ths, sell_ths, holds, cools, filters):
    r = run_one(train, buy, sell, hold, cool, tf)
    if best is None or r.total_return_pct > best[0]:
        best = (r.total_return_pct, buy, sell, hold, cool, tf, r)

_, buy, sell, hold, cool, tf, r_train = best
print(f"Best on TRAIN: buy={buy} sell={sell} hold={hold} cool={cool} filt={tf}")
print(f"  Train: {r_train.total_return_pct:+.2f}%  trades={r_train.total_trades}  MaxDD={r_train.max_drawdown_pct:.2f}%")

# Evaluate the SAME config on unseen test
r_test = run_one(test, buy, sell, hold, cool, tf)
print(f"\n=== OUT-OF-SAMPLE (test) ===")
print(f"  Strategy: {r_test.total_return_pct:+.2f}%  trades={r_test.total_trades}  "
      f"MaxDD={r_test.max_drawdown_pct:.2f}%  Sharpe={r_test.sharpe_ratio:.2f}  Win={r_test.win_rate:.1f}%")
print(f"  B&H:      {bh_test:+.2f}%")
print(f"  Verdict: {'策略贏' if r_test.total_return_pct > bh_test else 'Buy&Hold贏'}")

# Robustness: how do the top-5 train configs do on test?
print(f"\n=== Top-5 train configs on TEST (robustness check) ===")
all_train = []
for buy, sell, hold, cool, tf in product(buy_ths, sell_ths, holds, cools, filters):
    r = run_one(train, buy, sell, hold, cool, tf)
    all_train.append((r.total_return_pct, buy, sell, hold, cool, tf))
all_train.sort(key=lambda x: x[0], reverse=True)

test_rets = []
for _, buy, sell, hold, cool, tf in all_train[:5]:
    rt = run_one(test, buy, sell, hold, cool, tf)
    test_rets.append(rt.total_return_pct)
    print(f"  buy={buy} sell={sell} filt={tf}: train_sel -> test {rt.total_return_pct:+.2f}% "
          f"({rt.total_trades} trades)")

avg_test = sum(test_rets) / len(test_rets)
print(f"\nTop-5 avg test return: {avg_test:+.2f}% vs B&H test {bh_test:+.2f}%")
