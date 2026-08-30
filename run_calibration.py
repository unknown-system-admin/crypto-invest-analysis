#!/usr/bin/env python3
"""Threshold calibration script with grid search for optimal buy/sell thresholds."""
import sys
sys.path.insert(0, ".")

import warnings
warnings.filterwarnings("ignore")

import ccxt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from itertools import product

from feature_engine.builder import build_feature_matrix
from feature_engine.momentum import momentum_score, momentum_delta, momentum_acceleration
from feature_engine.indicators import compute_all_indicators
from backtest_engine.engine import BacktestEngine
from backtest_engine.rule_strategy import MomentumRuleStrategy


def fetch_data():
    """Fetch BTC/USDT daily data from OKX (365 days)."""
    print("Fetching data from OKX...")
    exchange = ccxt.okx()
    since = int((datetime.now() - timedelta(days=365)).timestamp() * 1000)
    ohlcv = exchange.fetch_ohlcv("BTC/USDT", "1d", since=since, limit=365)

    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    print(f"  Data range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"  Rows: {len(df)}")
    return df


def build_features(df):
    """Build feature matrix with momentum scores."""
    print("\nBuilding feature matrix...")
    indicators = compute_all_indicators(df)
    indicators["close"] = df["close"]

    momentum = momentum_score(indicators)
    delta = momentum_delta(momentum)
    acceleration = momentum_acceleration(delta)

    indicators["momentum_score"] = momentum
    indicators["momentum_delta"] = delta
    indicators["momentum_acceleration"] = acceleration

    valid_idx = indicators.dropna().index
    features = indicators.loc[valid_idx]
    print(f"  Valid rows after indicators: {len(features)}")
    return features


def analyze_momentum_distribution(features):
    """Calculate momentum score statistics."""
    print("\nAnalyzing momentum distribution...")
    scores = features["momentum_score"]

    stats = {
        "mean": round(scores.mean(), 4),
        "std": round(scores.std(), 4),
        "min": round(scores.min(), 4),
        "max": round(scores.max(), 4),
        "25%": round(scores.quantile(0.25), 4),
        "50%": round(scores.quantile(0.50), 4),
        "75%": round(scores.quantile(0.75), 4),
        "skewness": round(scores.skew(), 4),
        "kurtosis": round(scores.kurtosis(), 4),
    }

    print(f"  Mean: {stats['mean']}")
    print(f"  Std:  {stats['std']}")
    print(f"  Min:  {stats['min']}")
    print(f"  Max:  {stats['max']}")
    print(f"  25%:  {stats['25%']}")
    print(f"  50%:  {stats['50%']}")
    print(f"  75%:  {stats['75%']}")

    return stats


def run_backtest(features, buy_threshold, sell_threshold):
    """Run backtest with given thresholds and return result."""
    strategy = MomentumRuleStrategy(
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold
    )
    engine = BacktestEngine(
        strategy=strategy,
        initial_capital=10000.0,
        fee_rate=0.001,
        slippage=0.0005,
        max_position_pct=25.0,
        max_daily_trades=10,
        symbol="BTC/USDT",
        timeframe="1d"
    )
    return engine.run(features)


def grid_search_thresholds(features):
    """Grid search over buy/sell threshold combinations."""
    print("\nRunning grid search...")
    buy_range = np.arange(0.05, 0.26, 0.01)
    sell_range = np.arange(-0.25, -0.04, 0.01)

    buy_values = [round(x, 2) for x in buy_range]
    sell_values = [round(x, 2) for x in sell_range]

    total = len(buy_values) * len(sell_values)
    print(f"  Testing {len(buy_values)} buy x {len(sell_values)} sell = {total} combinations")

    results = []
    completed = 0

    for buy_thresh, sell_thresh in product(buy_values, sell_values):
        try:
            bt = run_backtest(features, buy_thresh, sell_thresh)
            results.append({
                "buy_threshold": buy_thresh,
                "sell_threshold": sell_thresh,
                "total_trades": bt.total_trades,
                "total_return_pct": bt.total_return_pct,
                "max_drawdown_pct": bt.max_drawdown_pct,
                "sharpe_ratio": bt.sharpe_ratio,
                "win_rate": bt.win_rate,
                "final_equity": bt.final_equity,
            })
        except Exception as e:
            print(f"  Error with buy={buy_thresh}, sell={sell_thresh}: {e}")

        completed += 1
        if completed % 100 == 0:
            print(f"  Progress: {completed}/{total} ({completed/total*100:.1f}%)")

    print(f"\n  Completed {len(results)} backtests")
    return results


def find_best_thresholds(results):
    """Rank results by Sharpe ratio and return top 10."""
    print("\nFinding best thresholds...")
    sorted_results = sorted(results, key=lambda x: x["sharpe_ratio"], reverse=True)

    best = sorted_results[0]
    print(f"\n  Best combination:")
    print(f"    Buy threshold:   {best['buy_threshold']}")
    print(f"    Sell threshold:  {best['sell_threshold']}")
    print(f"    Total return:    {best['total_return_pct']:.2f}%")
    print(f"    Sharpe ratio:    {best['sharpe_ratio']:.2f}")
    print(f"    Win rate:        {best['win_rate']:.1f}%")
    print(f"    Total trades:    {best['total_trades']}")
    print(f"    Max drawdown:    {best['max_drawdown_pct']:.2f}%")

    return sorted_results[:10]


def generate_report(momentum_stats, top_results, best):
    """Generate calibration report in markdown."""
    report = f"""# 閾值校準報告

生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 動能分數分布

| 統計量 | 數值 |
|--------|------|
| 平均值 | {momentum_stats['mean']:.4f} |
| 標準差 | {momentum_stats['std']:.4f} |
| 最小值 | {momentum_stats['min']:.4f} |
| 最大值 | {momentum_stats['max']:.4f} |
| 25% | {momentum_stats['25%']:.4f} |
| 50% | {momentum_stats['50%']:.4f} |
| 75% | {momentum_stats['75%']:.4f} |
| 偏度 | {momentum_stats['skewness']:.4f} |
| 峰度 | {momentum_stats['kurtosis']:.4f} |

## 最佳閾值組合 (Top 10)

| 排名 | Buy | Sell | 報酬% | Sharpe | 勝率 | 交易次數 | 最大回撤% |
|------|-----|------|-------|--------|------|----------|-----------|
"""
    for i, r in enumerate(top_results, 1):
        report += f"| {i} | {r['buy_threshold']:.2f} | {r['sell_threshold']:.2f} | {r['total_return_pct']:.2f}% | {r['sharpe_ratio']:.2f} | {r['win_rate']:.1f}% | {r['total_trades']} | {r['max_drawdown_pct']:.2f}% |\n"

    report += f"""
## 建議

- **最佳 buy threshold:** {best['buy_threshold']:.2f}
- **最佳 sell threshold:** {best['sell_threshold']:.2f}
- **預期報酬:** {best['total_return_pct']:.2f}%
- **預期夏普:** {best['sharpe_ratio']:.2f}
- **預期勝率:** {best['win_rate']:.1f}%
- **預期交易次數:** {best['total_trades']}
- **最大回撤:** {best['max_drawdown_pct']:.2f}%

## 動能分數範圍分析

基於歷史動能分數分布（均值 {momentum_stats['mean']:.4f}，標準差 {momentum_stats['std']:.4f}）：

- Buy threshold 建議範圍: 0.05 ~ 0.25（高於均值 1-2 個標準差）
- Sell threshold 建議範圍: -0.25 ~ -0.05（低於均值 1-2 個標準差）

## 使用方式

```python
from backtest_engine.rule_strategy import MomentumRuleStrategy

strategy = MomentumRuleStrategy(
    buy_threshold={best['buy_threshold']:.2f},
    sell_threshold={best['sell_threshold']:.2f}
)
```
"""
    return report


def main():
    print("=" * 60)
    print("CRYPTO INVESTMENT ANALYSIS - THRESHOLD CALIBRATION")
    print("=" * 60)

    # Fetch data
    df = fetch_data()

    # Build features
    features = build_features(df)

    # Analyze momentum distribution
    momentum_stats = analyze_momentum_distribution(features)

    # Grid search thresholds
    results = grid_search_thresholds(features)

    # Find best thresholds
    top_results = find_best_thresholds(results)
    best = top_results[0]

    # Generate report
    print("\n" + "=" * 60)
    print("GENERATING REPORT")
    print("=" * 60)

    report = generate_report(momentum_stats, top_results, best)

    report_path = "calibration_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nReport saved to: {report_path}")
    print("=" * 60)
    print("CALIBRATION COMPLETE")
    print("=" * 60)

    return best


if __name__ == "__main__":
    main()
