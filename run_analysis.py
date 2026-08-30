#!/usr/bin/env python3
"""Main analysis script for indicator effectiveness and ML feature selection."""
import sys
sys.path.insert(0, ".")

import ccxt
import pandas as pd
from datetime import datetime, timedelta
from feature_engine.builder import build_feature_matrix
from feature_engine.momentum import momentum_score, momentum_delta
from analyzers.momentum_analyzer import MomentumAnalyzer
from analyzers.feature_selector import FeatureSelector
from analyzers.report_generator import ReportGenerator


def main():
    print("=" * 60)
    print("指標分析系統")
    print("=" * 60)

    # 1. Fetch data
    print("\n[1/5] 從 OKX 獲取數據...")
    exchange = ccxt.okx()
    since = int((datetime.now() - timedelta(days=365)).timestamp() * 1000)
    ohlcv = exchange.fetch_ohlcv("BTC/USDT", "1d", since=since, limit=365)

    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    print(f"  數據範圍: {df.index[0].date()} 至 {df.index[-1].date()}")
    print(f"  數據筆數: {len(df)}")

    # 2. Build features
    print("\n[2/5] 計算技術指標...")
    features, labels = build_feature_matrix(df, n_bars=5)
    print(f"  特徵數量: {features.shape[1]}")
    print(f"  有效樣本: {features.shape[0]}")

    # 3. Momentum analysis
    print("\n[3/5] 動能指標有效性分析...")
    momentum_analyzer = MomentumAnalyzer()

    # Add momentum features to df for analysis
    df["momentum_score"] = features["momentum_score"]
    df["momentum_delta"] = features["momentum_delta"]

    correlation_result = momentum_analyzer.analyze_correlation(df, forward_days=5)
    print(f"  相關係數: {correlation_result['correlation']}")
    print(f"  P 值: {correlation_result['p_value']}")
    print(f"  顯著性: {'顯著' if correlation_result['significant'] else '不顯著'}")

    # 4. Threshold backtest
    print("\n[4/5] 閾值回測...")
    backtest_results = momentum_analyzer.backtest_thresholds(
        df,
        buy_thresholds=[0.2, 0.3, 0.4, 0.5],
        sell_thresholds=[-0.2, -0.3, -0.4, -0.5],
    )

    # Find best result
    best = max(backtest_results, key=lambda x: x.get("sharpe_ratio", 0))
    print(f"  最佳閾值: buy={best['buy_threshold']}, sell={best['sell_threshold']}")
    print(f"  報酬率: {best['total_return_pct']:.1f}%")
    print(f"  夏普比率: {best['sharpe_ratio']:.2f}")

    # 5. ML feature selection
    print("\n[5/5] ML 特徵篩選...")
    selector = FeatureSelector()

    # Prepare data for ML
    ml_features = features.drop(columns=["close"], errors="ignore")
    ml_features["binary_label"] = labels

    # Remove NaN
    ml_features = ml_features.dropna()

    feature_columns = [col for col in ml_features.columns if col != "binary_label"]

    ml_result = selector.train(
        ml_features,
        feature_columns=feature_columns,
        label_column="binary_label",
    )

    top_features = selector.get_top_features(n=10)
    print(f"  模型準確率: {ml_result['accuracy']:.4f}")
    print(f"  Top 10 指標: {', '.join(top_features)}")

    # Generate report
    print("\n生成分析報告...")
    generator = ReportGenerator()
    report = generator.full_report(
        momentum_result=correlation_result,
        feature_result=ml_result,
        backtest_results=backtest_results,
    )

    # Save report
    report_path = "analysis_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n報告已儲存至: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
