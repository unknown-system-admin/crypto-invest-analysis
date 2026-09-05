"""Phase 3: simplest possible regime benchmark.

close > SMA_200 -> hold BTC. close < SMA_200 -> flat (v1) or short (v2).
Cost: 0.15% (fee+slippage) per position change.
No engine, no momentum — just the regime switch.
"""

import pandas as pd
from data_cache import load_or_fetch
from feature_engine.builder import build_feature_matrix

df = load_or_fetch("BTC/USDT", "1d", limit=1000)
features, labels = build_feature_matrix(df, n_bars=5)

close = features["close"]
sma200 = features["SMA_200"]
daily_ret = close.pct_change().fillna(0)

COST = 0.0015  # per position swap

for name, pos in [
    ("Long/Flat", pd.Series(1.0, index=close.index).where(close > sma200, 0.0)),
    ("Long/Short", pd.Series(1.0, index=close.index).where(close > sma200, -1.0)),
]:
    pos_lagged = pos.shift(1).fillna(0.0)
    swap_cost = pos_lagged.diff().abs().fillna(0.0) * COST
    strat_ret = pos_lagged * daily_ret - swap_cost
    equity = (1 + strat_ret).cumprod()

    total = (equity.iloc[-1] - 1) * 100
    years = len(features) / 365.0

    # Split same as walk-forward: 60/40
    split = int(len(features) * 0.6)
    tr = equity.iloc[:split]
    te = equity.iloc[split:]
    tr_ret = (tr.iloc[-1] / tr.iloc[0] - 1) * 100
    te_ret = (te.iloc[-1] / te.iloc[0] - 1) * 100

    # MaxDD
    roll_peak = equity.cummax()
    maxdd = ((equity / roll_peak) - 1).min() * 100

    # Trades (swaps)
    swaps = int((pos.diff().abs() > 0).sum())

    print(f"{name}: total {total:+.2f}%  train {tr_ret:+.2f}%  test {te_ret:+.2f}%  "
          f"MaxDD {maxdd:.2f}%  swaps={swaps}")

# Reference
split = int(len(features) * 0.6)
bh_total = (close.iloc[-1] / close.iloc[0] - 1) * 100
bh_train = (close.iloc[:split].iloc[-1] / close.iloc[0] - 1) * 100
bh_test = (close.iloc[split:].iloc[-1] / close.iloc[split:].iloc[0] - 1) * 100
print(f"\nB&H:      total {bh_total:+.2f}%  train {bh_train:+.2f}%  test {bh_test:+.2f}%")
print(f"(Best momentum w/ filter: train +56.9%*  test +13.72%)")
