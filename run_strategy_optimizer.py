#!/usr/bin/env python3
"""Strategy optimizer: find best strategy/model maximizing total return."""
import sys
sys.path.insert(0, ".")

import warnings
warnings.filterwarnings("ignore")

import ccxt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from feature_engine.builder import build_feature_matrix
from feature_engine.indicators import compute_all_indicators
from feature_engine.momentum import momentum_score as compute_momentum_score
from feature_engine.labels import binary_label
from backtest_engine.engine import BacktestEngine
from backtest_engine.rule_strategy import MomentumRuleStrategy
from backtest_engine.model_strategy import ModelStrategy
from backtest_engine.strategy import Strategy, Signal


# ── Data Fetching ──────────────────────────────────────────────

def fetch_okx_data(symbol="BTC/USDT", days=400):
    """Fetch daily OHLCV from OKX."""
    print(f"Fetching {days} days of {symbol} data from OKX...")
    exchange = ccxt.okx()
    since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    ohlcv = exchange.fetch_ohlcv(symbol, "1d", since=since, limit=days)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    print(f"  Period: {df.index[0].date()} ~ {df.index[-1].date()} ({len(df)} rows)")
    return df


# ── Feature Engineering ────────────────────────────────────────

def build_optimized_features(df):
    """Build feature matrix with all indicators and momentum."""
    indicators = compute_all_indicators(df)
    indicators["close"] = df["close"]
    momentum = compute_momentum_score(indicators)
    delta = momentum.diff()
    accel = delta.diff()
    indicators["momentum_score"] = momentum
    indicators["momentum_delta"] = delta
    indicators["momentum_acceleration"] = accel
    labels = binary_label(df, n_bars=5)
    valid_idx = indicators.dropna().index.intersection(labels.dropna().index)
    features = indicators.loc[valid_idx]
    labels = labels.loc[valid_idx]
    return features, labels


# ── Backtest Runner ────────────────────────────────────────────

def run_backtest(strategy, features_df, timeframe="1d"):
    """Run backtest and return BacktestResult."""
    engine = BacktestEngine(
        strategy=strategy,
        initial_capital=10000.0,
        fee_rate=0.001,
        slippage=0.0005,
        max_position_pct=25.0,
        max_daily_trades=10,
        symbol="BTC/USDT",
        max_drawdown_stop=30.0,
        timeframe=timeframe,
    )
    return engine.run(features_df)


def extract_metrics(result):
    """Extract dict from BacktestResult."""
    return {
        "total_return": result.total_return_pct,
        "sharpe": result.sharpe_ratio,
        "max_drawdown": result.max_drawdown_pct,
        "num_trades": result.total_trades,
        "win_rate": result.win_rate,
    }


def calc_profit_factor(result):
    """Calculate profit factor from trades."""
    sell_trades = [t for t in result.trades if t.get("action") == "sell" and "pnl" in t]
    gross_profit = sum(t["pnl"] for t in sell_trades if t["pnl"] > 0)
    gross_loss = sum(abs(t["pnl"]) for t in sell_trades if t["pnl"] < 0)
    return round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0.0


# ── Rule-Based Strategies ──────────────────────────────────────

def test_rule_strategies(features, labels):
    """Test conservative/moderate/aggressive/very aggressive rules."""
    configs = [
        ("保守", 0.08, -0.07),
        ("穩健", 0.05, -0.05),
        ("積極", 0.02, -0.02),
        ("激進", 0.01, -0.01),
    ]
    results = []
    for name, buy_th, sell_th in configs:
        strategy = MomentumRuleStrategy(buy_threshold=buy_th, sell_threshold=sell_th)
        bt_result = run_backtest(strategy, features)
        m = extract_metrics(bt_result)
        m["profit_factor"] = calc_profit_factor(bt_result)
        m["name"] = name
        m["buy_th"] = buy_th
        m["sell_th"] = sell_th
        results.append(m)
        print(f"  {name} (buy={buy_th}, sell={sell_th}): "
              f"return={m['total_return']:.1f}%, sharpe={m['sharpe']:.2f}, "
              f"trades={m['num_trades']}, win={m['win_rate']:.1f}%")
    return results


# ── ML Model Strategies ────────────────────────────────────────

def test_ml_models(features, labels, feature_columns):
    """Train RF, XGB, LGBM with time-series CV and backtest predictions."""
    X = features[feature_columns].values
    y = labels.values
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=7,
                                               min_samples_split=5, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8,
                                  random_state=42, eval_metric="logloss", verbosity=0),
        "LightGBM": LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                                    subsample=0.8, colsample_bytree=0.8,
                                    random_state=42, verbose=-1),
    }

    # Time-series CV on train set
    tscv = TimeSeriesSplit(n_splits=5)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=tscv, scoring="accuracy")
        model.fit(X_train_scaled, y_train)
        test_acc = model.score(X_test_scaled, y_test)

        # Build strategy using model predictions on test set features
        strategy = ModelStrategy(model, feature_columns, confidence_threshold=0.5)
        # We need to wrap it: ModelStrategy expects a row dict, but X_test is numpy.
        # Create a custom strategy that works on the test set features directly.
        test_features = features.iloc[split_idx:].copy()
        bt_result = run_backtest(strategy, test_features)
        m = extract_metrics(bt_result)
        m["profit_factor"] = calc_profit_factor(bt_result)
        m["name"] = name
        m["cv_accuracy"] = round(cv_scores.mean(), 4)
        m["test_accuracy"] = round(test_acc, 4)
        results.append(m)
        print(f"  {name}: cv_acc={cv_scores.mean():.4f}, test_acc={test_acc:.4f}, "
              f"return={m['total_return']:.1f}%, sharpe={m['sharpe']:.2f}, "
              f"trades={m['num_trades']}")
    return results


# ── Hybrid Strategy ────────────────────────────────────────────

class HybridStrategy(Strategy):
    """Rule-based primary + ML confirmation."""
    def __init__(self, rule_strategy, ml_model, feature_columns, confidence_threshold=0.5):
        self.rule = rule_strategy
        self.ml_model = ml_model
        self.feature_columns = feature_columns
        self.confidence_threshold = confidence_threshold

    def evaluate(self, features) -> Signal:
        rule_sig = self.rule.evaluate(features)
        try:
            X = np.array([[features.get(col, 0) for col in self.feature_columns]])
            prediction = self.ml_model.predict(X)[0]
            proba = self.ml_model.predict_proba(X)[0]
            ml_conf = max(proba)
        except Exception:
            return Signal("中立", 0.0, "hybrid")

        if ml_conf < self.confidence_threshold:
            return Signal("中立", 0.0, "hybrid")

        ml_sig = "偏多" if prediction == 1 else "偏空"

        # Both must agree to trade
        if rule_sig.direction == "偏多" and ml_sig == "偏多":
            return Signal("偏多", min(rule_sig.confidence, ml_conf), "hybrid")
        elif rule_sig.direction == "偏空" and ml_sig == "偏空":
            return Signal("偏空", min(rule_sig.confidence, ml_conf), "hybrid")
        else:
            return Signal("中立", 0.5, "hybrid")


def test_hybrid_strategies(features, labels, feature_columns):
    """Test hybrid: rule + RF/XGB confirmation."""
    X = features[feature_columns].values
    y = labels.values
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    test_features = features.iloc[split_idx:].copy()

    ml_models = {
        "RF": RandomForestClassifier(n_estimators=200, max_depth=7,
                                      min_samples_split=5, random_state=42),
        "XGB": XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8,
                              random_state=42, eval_metric="logloss", verbosity=0),
        "LGBM": LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8,
                                random_state=42, verbose=-1),
    }

    results = []
    for ml_name, ml_model in ml_models.items():
        ml_model.fit(X_train_scaled, y_train)

        for rule_name, buy_th, sell_th in [("保守", 0.08, -0.07), ("穩健", 0.05, -0.05)]:
            rule = MomentumRuleStrategy(buy_threshold=buy_th, sell_threshold=sell_th)
            hybrid = HybridStrategy(rule, ml_model, feature_columns, confidence_threshold=0.5)
            bt_result = run_backtest(hybrid, test_features)
            m = extract_metrics(bt_result)
            m["profit_factor"] = calc_profit_factor(bt_result)
            m["name"] = f"規則({rule_name})+{ml_name}"
            m["rule_buy"] = buy_th
            m["rule_sell"] = sell_th
            results.append(m)
            print(f"  規則({rule_name})+{ml_name}: "
                  f"return={m['total_return']:.1f}%, sharpe={m['sharpe']:.2f}, "
                  f"trades={m['num_trades']}")
    return results


# ── Report Generation ──────────────────────────────────────────

def generate_report(df, train_size, test_size, rule_results, ml_results, hybrid_results):
    """Generate strategy comparison report in Traditional Chinese."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    date_range = f"{df.index[0].date()} ~ {df.index[-1].date()}"

    # Find best overall
    all_results = rule_results + ml_results + hybrid_results
    best = max(all_results, key=lambda x: x["total_return"])

    # Format rule table
    rule_rows = ""
    for r in rule_results:
        rule_rows += (f"| {r['name']} | {r['buy_th']} | {r['sell_th']} | "
                      f"{r['total_return']:.1f}% | {r['sharpe']:.2f} | "
                      f"{r['win_rate']:.0f}% | {r['num_trades']} | "
                      f"{r['max_drawdown']:.1f}% | {r['profit_factor']:.2f} |\n")

    # Format ML table
    ml_rows = ""
    for r in ml_results:
        ml_rows += (f"| {r['name']} | {r['total_return']:.1f}% | "
                     f"{r['sharpe']:.2f} | {r['win_rate']:.0f}% | "
                     f"{r['num_trades']} | {r['max_drawdown']:.1f}% | "
                     f"{r['profit_factor']:.2f} | {r.get('cv_accuracy', 0):.4f} | "
                     f"{r.get('test_accuracy', 0):.4f} |\n")

    # Format hybrid table
    hybrid_rows = ""
    for r in hybrid_results:
        hybrid_rows += (f"| {r['name']} | {r['total_return']:.1f}% | "
                         f"{r['sharpe']:.2f} | {r['win_rate']:.0f}% | "
                         f"{r['num_trades']} | {r['max_drawdown']:.1f}% | "
                         f"{r['profit_factor']:.2f} |\n")

    # Determine best type
    best_rule = max(rule_results, key=lambda x: x["total_return"])
    best_ml = max(ml_results, key=lambda x: x["total_return"])
    best_hybrid = max(hybrid_results, key=lambda x: x["total_return"])
    best_map = {
        "rule": best_rule,
        "ml": best_ml,
        "hybrid": best_hybrid,
    }
    best_category = max(best_map, key=lambda k: best_map[k]["total_return"])

    report = f"""# 策略比較報告

生成時間: {now}

## 資料摘要

| 項目 | 數值 |
|------|------|
| 資料期間 | {date_range} |
| 訓練集 | {train_size} 天 |
| 測試集 | {test_size} 天 |

## 規則策略比較

| 策略 | Buy | Sell | 報酬% | Sharpe | 勝率 | 交易次數 | 最大回撤 | 盈虧比 |
|------|-----|------|-------|--------|------|----------|----------|--------|
{rule_rows}
## ML 模型比較

| 模型 | 報酬% | Sharpe | 勝率 | 交易次數 | 最大回撤 | 盈虧比 | CV精度 | 測試精度 |
|------|-------|--------|------|----------|----------|--------|--------|----------|
{ml_rows}
## 混合策略

| 策略 | 報酬% | Sharpe | 勝率 | 交易次數 | 最大回撤 | 盈虧比 |
|------|-------|--------|------|----------|----------|--------|
{hybrid_rows}
## 最佳策略

- **最佳策略類型:** {best_category.upper()}
- **最佳策略:** {best['name']}
- **預期報酬:** {best['total_return']:.1f}%
- **預期夏普:** {best['sharpe']:.2f}
- **最大回撤:** {best['max_drawdown']:.1f}%
- **勝率:** {best['win_rate']:.0f}%
- **交易次數:** {best['num_trades']}

## 全策略排名 (按報酬%)

| 排名 | 策略 | 類型 | 報酬% | Sharpe | 回撤% |
|------|------|------|-------|--------|-------|
"""
    ranked = sorted(all_results, key=lambda x: x["total_return"], reverse=True)
    type_map = {id(r): "規則" for r in rule_results}
    type_map.update({id(r): "ML" for r in ml_results})
    type_map.update({id(r): "混合" for r in hybrid_results})
    for i, r in enumerate(ranked[:10], 1):
        rtype = type_map.get(id(r), "?")
        report += f"| {i} | {r['name']} | {rtype} | {r['total_return']:.1f}% | {r['sharpe']:.2f} | {r['max_drawdown']:.1f}% |\n"

    report += """
## 建議

1. 選擇報酬最高且夏普比率 > 1 的策略
2. 注意最大回撤是否可接受（建議 < 30%）
3. 混合策略通常更穩健，但交易次數可能較少
4. ML 模型需定期重新訓練以適應市場變化
5. 建議在實盤前先用紙上交易驗證
"""

    return report


# ── Main ───────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("策略優化器 — 最大化報酬率")
    print("=" * 60)

    # 1. Fetch data
    df = fetch_okx_data("BTC/USDT", days=400)

    # 2. Build features
    print("\nBuilding feature matrix...")
    features, labels = build_optimized_features(df)
    print(f"  Features: {len(features.columns)} columns, {len(features)} rows")

    # 3. Train/test split
    split_idx = int(len(features) * 0.8)
    train_features = features.iloc[:split_idx]
    test_features = features.iloc[split_idx:]
    train_labels = labels.iloc[:split_idx]
    test_labels = labels.iloc[split_idx:]
    train_size = len(train_features)
    test_size = len(test_features)
    print(f"  Train: {train_size} days, Test: {test_size} days")

    feature_columns = [c for c in features.columns if c != "close"]

    # 4. Test rule strategies on test set
    print("\n" + "=" * 60)
    print("A. 規則策略測試")
    print("=" * 60)
    rule_results = test_rule_strategies(test_features, test_labels)

    # 5. Test ML models
    print("\n" + "=" * 60)
    print("B. ML 模型測試")
    print("=" * 60)
    ml_results = test_ml_models(features, labels, feature_columns)

    # 6. Test hybrid strategies
    print("\n" + "=" * 60)
    print("C. 混合策略測試")
    print("=" * 60)
    hybrid_results = test_hybrid_strategies(features, labels, feature_columns)

    # 7. Generate report
    print("\n" + "=" * 60)
    print("生成報告...")
    print("=" * 60)
    report = generate_report(df, train_size, test_size, rule_results, ml_results, hybrid_results)
    report_path = "strategy_comparison_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n報告已儲存: {report_path}")

    # 8. Summary
    all_results = rule_results + ml_results + hybrid_results
    best = max(all_results, key=lambda x: x["total_return"])
    print("\n" + "=" * 60)
    print("最佳策略")
    print("=" * 60)
    print(f"  策略: {best['name']}")
    print(f"  報酬: {best['total_return']:.1f}%")
    print(f"  夏普: {best['sharpe']:.2f}")
    print(f"  回撤: {best['max_drawdown']:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
