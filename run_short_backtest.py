#!/usr/bin/env python
"""Run short selling backtest comparison."""

import joblib
import pandas as pd
from datetime import datetime
from data_cache import load_or_fetch
from feature_engine.builder import build_feature_matrix
from backtest_engine.engine import BacktestEngine
from backtest_engine.rule_strategy import MomentumRuleStrategy
from backtest_engine.short_strategy import MomentumShortStrategy, MLShortStrategy


def load_model_and_scaler():
    model = joblib.load("model_rf_optimized.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler


def run_backtest(strategy, features_df, label, timeframe="1h"):
    engine = BacktestEngine(
        strategy=strategy,
        initial_capital=10000,
        timeframe=timeframe,
    )
    result = engine.run(features_df)
    return {
        "strategy": label,
        "type": "多空" if "Short" in label or "short" in label else "做多",
        "total_return_pct": result.total_return_pct,
        "sharpe_ratio": result.sharpe_ratio,
        "max_drawdown_pct": result.max_drawdown_pct,
        "total_trades": result.total_trades,
        "win_rate": result.win_rate,
    }


def generate_report(results, output_path="short_strategy_report.md"):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Find best strategy by Sharpe
    best = max(results, key=lambda x: x["sharpe_ratio"])

    lines = [
        "# 做空策略回測報告",
        "",
        f"生成時間: {now}",
        "",
        "## 策略比較",
        "",
        "| 策略 | 類型 | 總報酬% | Sharpe | 最大回撤% | 交易次數 | 勝率 |",
        "|------|------|---------|--------|-----------|----------|------|",
    ]

    for r in results:
        lines.append(
            f"| {r['strategy']} | {r['type']} | {r['total_return_pct']:.1f}% "
            f"| {r['sharpe_ratio']:.2f} | {r['max_drawdown_pct']:.1f}% "
            f"| {r['total_trades']} | {r['win_rate']:.0f}% |"
        )

    lines.extend([
        "",
        "## 最佳策略",
        "",
        f"- **策略名稱:** {best['strategy']}",
        f"- **策略類型:** {best['type']}",
        f"- **總報酬:** {best['total_return_pct']:.1f}%",
        f"- **Sharpe:** {best['sharpe_ratio']:.2f}",
        f"- **最大回撤:** {best['max_drawdown_pct']:.1f}%",
        "",
    ])

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    return output_path


def main():
    print("Loading data...")
    df = load_or_fetch("BTC/USDT", "1h", limit=8000)
    print(f"Data: {len(df)} candles from {df.index[0]} to {df.index[-1]}")

    print("Building features...")
    features_df, labels = build_feature_matrix(df)
    print(f"Features: {len(features_df)} rows, {len(features_df.columns)} columns")

    model, scaler = load_model_and_scaler()
    feature_cols = ["SMA_20", "SMA_50", "RSI", "MACD", "ATR", "MFI", "OBV"]

    results = []

    # 1. Momentum Long-only
    print("Running Momentum Long...")
    strat = MomentumRuleStrategy(buy_threshold=0.08, sell_threshold=-0.07)
    results.append(run_backtest(strat, features_df, "Momentum Long"))

    # 2. Momentum Long/Short
    print("Running Momentum Long/Short...")
    strat = MomentumShortStrategy(buy_threshold=0.01, sell_threshold=-0.01)
    results.append(run_backtest(strat, features_df, "Momentum Long/Short"))

    # 3. ML Long-only
    print("Running ML Long...")
    from backtest_engine.model_strategy import ModelStrategy
    strat = ModelStrategy(model, feature_cols, confidence_threshold=0.5)
    results.append(run_backtest(strat, features_df, "ML Long"))

    # 4. ML Long/Short
    print("Running ML Long/Short...")
    strat = MLShortStrategy(model, scaler, threshold=0.5)
    results.append(run_backtest(strat, features_df, "ML Long/Short"))

    # Generate report
    report_path = generate_report(results)
    print(f"\nReport saved to {report_path}")

    # Print summary
    print("\n=== Results ===")
    for r in results:
        print(f"{r['strategy']:25s} | Return: {r['total_return_pct']:7.1f}% | Sharpe: {r['sharpe_ratio']:5.2f} | MaxDD: {r['max_drawdown_pct']:5.1f}% | Trades: {r['total_trades']:3d} | WinRate: {r['win_rate']:.0f}%")


if __name__ == "__main__":
    main()
