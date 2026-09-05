"""Long-period multi-fold walk-forward validation for BTC and SOL.

Design:
- 3 expanding-window folds; test windows are the last 45% of data, in 3 chunks.
- Per fold: grid-select best config on train (unseen test never touched),
  then evaluate on test. Also evaluate the FIXED BTC config everywhere
  (transferability), plus B&H on each test window.

Grid selection is per-asset (user allows different params per asset).
"""

from itertools import product
from data_cache import load_from_cache
from feature_engine.builder import build_feature_matrix
from backtest_engine.engine import BacktestEngine
from backtest_engine.rule_strategy import MomentumRuleStrategy

INITIAL = 10000.0
FIXED = dict(buy=0.05, sell=-0.30, hold=0, cool=3, tf=True)

GRID = dict(
    buy_ths=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    sell_ths=[-0.05, -0.10, -0.15, -0.20, -0.25, -0.30],
    holds=[0, 5, 10],
    cools=[0, 3],
    filters=[False, True],
)


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


def bh(df_part):
    return (df_part["close"].iloc[-1] / df_part["close"].iloc[0] - 1) * 100


def validate(symbol):
    df = load_from_cache(symbol, "1d")
    features, _ = build_feature_matrix(df, n_bars=5)
    n = len(features)
    print(f"\n{'='*78}")
    print(f"{symbol}: {n} usable bars, {features.index[0].date()} -> {features.index[-1].date()}")

    # Folds: expanding train, 3 sequential test chunks covering last 45%
    bounds = [int(n * f) for f in (0.55, 0.70, 0.85)]
    folds = []
    start = None
    for b in bounds + [n]:
        folds.append((start, b))
        start = b
    folds = [(int(n * 0.55), int(n * 0.70)), (int(n * 0.70), int(n * 0.85)), (int(n * 0.85), n)]

    print(f"Test windows: " + ", ".join(
        f"{features.index[a].date()}->{features.index[b-1].date()}({b-a}b)" for a, b in folds))

    agg_sel, agg_fix, agg_bh = [], [], []
    print(f"\n{'Fold':<10}{'Test period':<24}{'Sel cfg':<34}{'Sel%':>8}{'Fixed%':>8}{'B&H%':>8}{'Trades':>8}")
    print("-" * 100)

    for k, (a, b) in enumerate(folds, 1):
        train = features.iloc[:a]
        test = features.iloc[a:b]
        period = f"{test.index[0].date()}~{test.index[-1].date()}"

        # Grid select on train
        best = None
        for buy, sell, hold, cool, tf in product(GRID["buy_ths"], GRID["sell_ths"],
                                                  GRID["holds"], GRID["cools"], GRID["filters"]):
            r = run_one(train, buy, sell, hold, cool, tf)
            if best is None or r.total_return_pct > best[0]:
                best = (r.total_return_pct, buy, sell, hold, cool, tf)

        _, buy, sell, hold, cool, tf = best
        r_sel = run_one(test, buy, sell, hold, cool, tf)
        r_fix = run_one(test, **{**FIXED, "tf": FIXED["tf"]})
        bh_ret = bh(test)

        cfg = f"b{buy} s{sell} h{hold} c{cool} f{int(tf)}"
        print(f"Fold{k:<5}{period:<24}{cfg:<34}{r_sel.total_return_pct:>7.2f}%"
              f"{r_fix.total_return_pct:>7.2f}%{bh_ret:>7.2f}%{r_sel.total_trades:>8}")

        agg_sel.append(r_sel.total_return_pct)
        agg_fix.append(r_fix.total_return_pct)
        agg_bh.append(bh_ret)

    # Aggregate: compound test-window returns
    def comp(rets):
        tot = 1.0
        for r in rets:
            tot *= (1 + r / 100)
        return (tot - 1) * 100

    print(f"\nAggregate over 3 test folds (compounded):")
    print(f"  Grid-selected: {comp(agg_sel):+.2f}%   Fixed BTC cfg: {comp(agg_fix):+.2f}%   B&H: {comp(agg_bh):+.2f}%")
    wins = sum(1 for s, b in zip(agg_sel, agg_bh) if s > b)
    print(f"  Grid-selected beat B&H in {wins}/3 folds")
    return comp(agg_sel), comp(agg_fix), comp(agg_bh)


print("Multi-fold walk-forward validation (train:select -> test:evaluate)")
btc = validate("BTC/USDT")
sol = validate("SOL/USDT")

print(f"\n{'='*78}")
print("SUMMARY (compounded OOS test returns, last 45% of data)")
print(f"  BTC: grid-selected {btc[0]:+.2f}% | fixed cfg {btc[1]:+.2f}% | B&H {btc[2]:+.2f}%")
print(f"  SOL: grid-selected {sol[0]:+.2f}% | fixed cfg (transferred) {sol[1]:+.2f}% | B&H {sol[2]:+.2f}%")
