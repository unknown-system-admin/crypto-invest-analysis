"""Diagnose: why does the fixed config lose in the bear fold despite shorting?

Look at actual position mix (long/short/flat days), short frequency,
and the momentum_score distribution to see how often the -0.30 gate triggers.
"""

from data_cache import load_from_cache
from feature_engine.builder import build_feature_matrix
from backtest_engine.engine import BacktestEngine
from backtest_engine.rule_strategy import MomentumRuleStrategy

INITIAL = 10000.0
CFG = dict(buy=0.05, sell=-0.30, hold=0, cool=3, tf=True)

for symbol in ["BTC/USDT", "SOL/USDT"]:
    df = load_from_cache(symbol, "1d")
    features, _ = build_feature_matrix(df, n_bars=5)
    n = len(features)

    # Fold2 (the bear): 70%-85%
    a, b = int(n * 0.70), int(n * 0.85)
    test = features.iloc[a:b]
    print(f"\n{'='*72}")
    print(f"{symbol} Fold2 bear: {test.index[0].date()} -> {test.index[-1].date()} ({len(test)} bars)")

    engine = BacktestEngine(
        strategy=MomentumRuleStrategy(buy_threshold=CFG["buy"], sell_threshold=CFG["sell"]),
        initial_capital=INITIAL, timeframe="1d", max_position_pct=95,
        max_drawdown_stop=50, trend_filter=CFG["tf"],
        min_holding_bars=CFG["hold"], cooldown_bars=CFG["cool"],
    )
    r = engine.run(test)

    # Trade log
    print(f"\nReturn {r.total_return_pct:+.2f}%, {r.total_trades} entries:")
    for t in r.trades:
        side = {"buy": "LONG", "short_sell": "SHORT", "sell": "close long", "cover": "close short"}.get(t["action"], t["action"])
        pnl = f" pnl={t['pnl']:+.0f}" if "pnl" in t else ""
        print(f"  {test.index[len(r.trades)] if False else ''}{side:<11} @ {t['price']:,.0f}{pnl}")

    # Position mix: simulate signals across the fold
    ms = test["momentum_score"].dropna()
    print(f"\nmomentum_score stats: min {ms.min():.2f}  median {ms.median():.2f}  "
          f"max {ms.max():.2f}  std {ms.std():.2f}")
    print(f"bars with score < -0.30 (would short): {(ms < -0.30).sum()}/{len(ms)}")
    print(f"bars with score < -0.05 (would exit long / weak short): {(ms < -0.05).sum()}/{len(ms)}")
    print(f"bars with score > 0.05: {(ms > 0.05).sum()}/{len(ms)}")
    print(f"bars close > SMA_200 (long regime): {(test['close'] > test['SMA_200']).sum()}/{len(test)}")
    print(f"bars close < SMA_200 (short regime): {(test['close'] < test['SMA_200']).sum()}/{len(test)}")