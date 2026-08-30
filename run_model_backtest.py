#!/usr/bin/env python
"""Model backtesting script comparing RF, XGB, and rule-based strategies."""

import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from data_cache import load_from_cache
from train_model import build_features, compute_local_indicators, compute_momentum_score
from backtest_engine.engine import BacktestEngine
from backtest_engine.model_strategy import ModelStrategy
from backtest_engine.rule_strategy import MomentumRuleStrategy


# ── Configuration ──────────────────────────────────────────────────────

SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
INITIAL_CAPITAL = 10000.0
FEE_RATE = 0.001
SLIPPAGE = 0.0005

# Model paths
RF_MODEL_PATH = "model_rf_optimized.pkl"
XGB_MODEL_PATH = "model_xgb_optimized.pkl"
SCALER_PATH = "scaler.pkl"


# ── Helper Functions ───────────────────────────────────────────────────

def load_models():
    """Load optimized RF and XGB models."""
    models = {}
    if Path(RF_MODEL_PATH).exists():
        models["RF"] = joblib.load(RF_MODEL_PATH)
        print(f"  Loaded RF model from {RF_MODEL_PATH}")
    else:
        print(f"  Warning: {RF_MODEL_PATH} not found")

    if Path(XGB_MODEL_PATH).exists():
        models["XGB"] = joblib.load(XGB_MODEL_PATH)
        print(f"  Loaded XGB model from {XGB_MODEL_PATH}")
    else:
        print(f"  Warning: {XGB_MODEL_PATH} not found")

    return models


def load_scaler():
    """Load the fitted scaler."""
    if Path(SCALER_PATH).exists():
        return joblib.load(SCALER_PATH)
    return None


def prepare_features(df: pd.DataFrame, scaler, feature_columns: list) -> pd.DataFrame:
    """Prepare features for backtesting."""
    # Build indicators
    indicators = compute_local_indicators(df)

    # Compute momentum
    indicators["close"] = df["close"]
    indicators["momentum_score"] = compute_momentum_score(indicators)
    indicators["momentum_delta"] = indicators["momentum_score"].diff()

    # Drop NaN rows
    indicators = indicators.dropna()

    # Scale features if scaler is available
    if scaler is not None:
        X = indicators[feature_columns]
        X_scaled = pd.DataFrame(
            scaler.transform(X),
            index=X.index,
            columns=feature_columns,
        )
        indicators[feature_columns] = X_scaled

    return indicators


def run_backtest(features: pd.DataFrame, strategy, name: str) -> dict:
    """Run a single backtest and return results."""
    engine = BacktestEngine(
        strategy=strategy,
        initial_capital=INITIAL_CAPITAL,
        fee_rate=FEE_RATE,
        slippage=SLIPPAGE,
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
    )
    result = engine.run(features)
    return {
        "name": name,
        "total_trades": result.total_trades,
        "final_equity": result.final_equity,
        "total_return_pct": result.total_return_pct,
        "max_drawdown_pct": result.max_drawdown_pct,
        "sharpe_ratio": result.sharpe_ratio,
        "win_rate": result.win_rate,
        "equity_curve": result.equity_curve,
    }


def generate_report(results: list, df: pd.DataFrame) -> str:
    """Generate markdown report comparing strategies."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    date_range = f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}"

    # Find best strategy
    best = max(results, key=lambda x: x["total_return_pct"])
    best_sharpe = max(results, key=lambda x: x["sharpe_ratio"])

    # Build comparison table
    rows = []
    for r in results:
        rows.append(f"| {r['name']} | {r['total_trades']} | ${r['final_equity']:,.2f} | "
                     f"{r['total_return_pct']:+.2f}% | {r['max_drawdown_pct']:.2f}% | "
                     f"{r['sharpe_ratio']:.2f} | {r['win_rate']:.1f}% |")

    comparison_table = "\n".join(rows)

    report = f"""# 模型回測報告

生成時間: {now}

## 回測配置
| 參數 | 數值 |
|------|------|
| 交易對 | {SYMBOL} |
| 時間框架 | {TIMEFRAME} |
| 初始資金 | ${INITIAL_CAPITAL:,.2f} |
| 手續費率 | {FEE_RATE * 100:.1f}% |
| 滑點 | {SLIPPAGE * 100:.2f}% |
| 數據範圍 | {date_range} |
| 數據筆數 | {len(df)} |

## 策略比較
| 策略 | 交易次數 | 最終資金 | 總收益率 | 最大回撤 | 夏普比率 | 勝率 |
|------|----------|----------|----------|----------|----------|------|
{comparison_table}

## 關鍵發現

### 最佳總收益
**{best['name']}** 策略獲得最高總收益率: **{best['total_return_pct']:+.2f}%**

### 最佳風險調整收益
**{best_sharpe['name']}** 策略獲得最高夏普比率: **{best_sharpe['sharpe_ratio']:.2f}**

### 策略分析

"""
    # Add analysis for each strategy
    for r in results:
        if r["name"] == best["name"]:
            report += f"- **{r['name']}**: 表現最佳，總收益 {r['total_return_pct']:+.2f}%，"
            report += f"最大回撤 {r['max_drawdown_pct']:.2f}%\n"
        else:
            report += f"- **{r['name']}**: 總收益 {r['total_return_pct']:+.2f}%，"
            report += f"最大回撤 {r['max_drawdown_pct']:.2f}%\n"

    report += f"""
## 結論

基於歷史數據回測，**{best['name']}** 策略在回測期間表現最佳。

⚠️ **注意**: 歷史回測結果不代表未來表現。實際交易需考慮市場變化、流動性等因素。
"""
    return report


# ── Main ───────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("模型回測腳本")
    print("=" * 60)

    # 1. Load cached data
    print("\n[1/5] 載入快取數據...")
    df = load_from_cache(SYMBOL, TIMEFRAME)
    if df is None or len(df) == 0:
        print("  錯誤: 無法載入快取數據，請先執行 data_cache.py")
        sys.exit(1)
    print(f"  載入 {len(df)} 根K線: {df.index[0]} -> {df.index[-1]}")

    # 2. Load models
    print("\n[2/5] 載入優化模型...")
    models = load_models()
    if not models:
        print("  錯誤: 無法載入任何模型")
        sys.exit(1)

    scaler = load_scaler()
    if scaler:
        print(f"  載入 Scaler")

    # 3. Prepare features
    print("\n[3/5] 準備特徵...")
    # Feature columns used during training (must match scaler.feature_names_in_)
    feature_columns = ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'ATR', 'MFI', 'OBV', 'close', 'momentum_score', 'momentum_delta']
    print(f"  使用 {len(feature_columns)} 個特徵: {', '.join(feature_columns)}")

    features = prepare_features(df, scaler, feature_columns)
    print(f"  準備 {len(features)} 個有效樣本")

    # 4. Run backtests
    print("\n[4/5] 執行回測...")

    # Add momentum score for rule-based strategy
    indicators = compute_local_indicators(df)
    indicators["close"] = df["close"]
    indicators["momentum_score"] = compute_momentum_score(indicators)
    indicators["momentum_delta"] = indicators["momentum_score"].diff()
    features_with_momentum = features.copy()
    features_with_momentum["momentum_score"] = indicators["momentum_score"]
    features_with_momentum["momentum_delta"] = indicators["momentum_delta"]
    features_with_momentum = features_with_momentum.dropna()

    results = []

    # Rule-based strategy
    print("  執行 Momentum Rule 策略...")
    rule_strategy = MomentumRuleStrategy(buy_threshold=0.08, sell_threshold=-0.07)
    results.append(run_backtest(features_with_momentum, rule_strategy, "Momentum Rule"))

    # Model strategies
    for name, model in models.items():
        print(f"  執行 {name} 策略...")
        strategy = ModelStrategy(model, feature_columns, confidence_threshold=0.55)
        results.append(run_backtest(features, strategy, name))

    # 5. Generate report
    print("\n[5/5] 生成報告...")
    report = generate_report(results, df)

    report_path = "model_backtest_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  報告已保存至 {report_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("回測結果摘要")
    print("=" * 60)
    for r in results:
        print(f"  {r['name']:15s}: 收益 {r['total_return_pct']:+7.2f}% | "
              f"回撤 {r['max_drawdown_pct']:5.2f}% | "
              f"夏普 {r['sharpe_ratio']:5.2f}")

    print("\n✅ 回測完成!")

    return results


if __name__ == "__main__":
    main()
