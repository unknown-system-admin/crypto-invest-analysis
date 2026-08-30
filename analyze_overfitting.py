#!/usr/bin/env python
"""Overfitting analysis for crypto prediction models.

Generates learning curves, cross-validation results, feature importance
stability, sample size analysis, and a comprehensive markdown report.
"""

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, learning_curve, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

from data_cache import load_or_fetch


# ── Feature Engineering (same as train_model.py) ─────────────────────


def compute_local_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    sma_20 = ta.trend.sma_indicator(close, window=20)
    sma_50 = ta.trend.sma_indicator(close, window=50)
    rsi = ta.momentum.rsi(close, window=14)
    macd_obj = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    macd = macd_obj.macd()
    atr = ta.volatility.average_true_range(high, low, close, window=14)
    mfi = ta.volume.money_flow_index(high, low, close, volume, window=14)
    obv = ta.volume.on_balance_volume(close, volume)

    return pd.DataFrame({
        "SMA_20": sma_20,
        "SMA_50": sma_50,
        "RSI": rsi,
        "MACD": macd,
        "ATR": atr,
        "MFI": mfi,
        "OBV": obv,
    }, index=df.index)


def compute_momentum_score(df: pd.DataFrame) -> pd.Series:
    rsi_norm = (df["RSI"] - 50) / 50
    macd_norm = np.tanh(df["MACD"] / df["MACD"].std())
    sma20_norm = (df["close"] - df["SMA_20"]) / df["SMA_20"]
    sma50_norm = (df["close"] - df["SMA_50"]) / df["SMA_50"]
    score = 0.3 * rsi_norm + 0.1 * macd_norm + 0.4 * sma20_norm + 0.2 * sma50_norm
    return score.clip(-1, 1)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    indicators = compute_local_indicators(df)
    indicators["close"] = df["close"]
    indicators["momentum_score"] = compute_momentum_score(indicators)
    indicators["momentum_delta"] = indicators["momentum_score"].diff()
    return indicators


def generate_labels(df: pd.DataFrame, n_bars: int = 5) -> pd.Series:
    future = df["close"].shift(-n_bars)
    ret = ((future - df["close"]) / df["close"]) * 100
    labels = (ret > 0).astype(float)
    labels[ret.isna()] = np.nan
    return labels


def prepare_data(symbol="BTC/USDT", timeframe="1h", limit=8000):
    df = load_or_fetch(symbol, timeframe, limit=limit)
    features = build_features(df)
    labels = generate_labels(df)

    valid = features.dropna().index.intersection(labels.dropna().index)
    X = features.loc[valid]
    y = labels.loc[valid]

    return X, y, df


# ── Analysis Functions ────────────────────────────────────────────────


def learning_curve_analysis(X, y):
    """Compute learning curves for train and CV scores."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(
        n_estimators=100, max_depth=7, min_samples_split=5,
        min_samples_leaf=2, random_state=42, n_jobs=-1
    )

    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, X_scaled, y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=TimeSeriesSplit(n_splits=5),
        scoring="accuracy",
        n_jobs=-1,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    gap = train_mean - val_mean
    overfit_idx = int(np.argmax(gap > 0.15))

    results = {
        "train_sizes": train_sizes_abs.tolist(),
        "train_mean": train_mean.tolist(),
        "train_std": train_std.tolist(),
        "val_mean": val_mean.tolist(),
        "val_std": val_std.tolist(),
        "max_gap": float(gap.max()),
        "final_gap": float(gap[-1]),
        "overfit_at_index": overfit_idx,
        "best_val_score": float(val_mean.max()),
    }
    return results


def cross_validation_analysis(X, y, n_splits=5):
    """Time-series cross-validation with multiple metrics."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(
        n_estimators=100, max_depth=7, min_samples_split=5,
        min_samples_leaf=2, random_state=42, n_jobs=-1
    )

    tscv = TimeSeriesSplit(n_splits=n_splits)

    acc_scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring="accuracy")
    f1_scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring="f1")

    fold_details = []
    for i, (train_idx, test_idx) in enumerate(tscv.split(X_scaled)):
        fold_details.append({
            "fold": i + 1,
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "acc": float(acc_scores[i]),
            "f1": float(f1_scores[i]),
        })

    return {
        "n_splits": n_splits,
        "acc_mean": float(acc_scores.mean()),
        "acc_std": float(acc_scores.std()),
        "f1_mean": float(f1_scores.mean()),
        "f1_std": float(f1_scores.std()),
        "acc_scores": acc_scores.tolist(),
        "f1_scores": f1_scores.tolist(),
        "fold_details": fold_details,
    }


def feature_importance_stability(X, y, n_runs=5):
    """Train model multiple times with different random seeds and check stability."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    importances_list = []
    for seed in range(n_runs):
        model = RandomForestClassifier(
            n_estimators=100, max_depth=7, min_samples_split=5,
            min_samples_leaf=2, random_state=seed, n_jobs=-1
        )
        model.fit(X_scaled, y)
        importances_list.append(model.feature_importances_)

    imp_array = np.array(importances_list)
    mean_imp = imp_array.mean(axis=0)
    std_imp = imp_array.std(axis=0)
    cv = np.where(mean_imp > 0, std_imp / mean_imp, 0)

    feature_rankings = []
    for i, feat in enumerate(X.columns):
        avg_rank = np.argsort(-mean_imp)[i]
        feature_rankings.append({
            "feature": feat,
            "mean_importance": float(mean_imp[i]),
            "std_importance": float(std_imp[i]),
            "cv": float(cv[i]),
            "avg_rank": int(avg_rank),
        })

    feature_rankings.sort(key=lambda x: x["avg_rank"])

    stable_features = sum(1 for f in feature_rankings if f["cv"] < 0.5)

    return {
        "n_runs": n_runs,
        "features": feature_rankings,
        "stable_count": stable_features,
        "total_features": len(X.columns),
        "stability_ratio": stable_features / len(X.columns) if X.columns.size > 0 else 0,
    }


def sample_size_analysis(X, y):
    """Analyze performance across different training set sizes."""
    fractions = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    results = []

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(
        n_estimators=100, max_depth=7, min_samples_split=5,
        min_samples_leaf=2, random_state=42, n_jobs=-1
    )

    for frac in fractions:
        n = int(len(X) * frac)
        if n < 50:
            continue

        X_sub = X_scaled[:n]
        y_sub = y.values[:n]

        tscv = TimeSeriesSplit(n_splits=3)
        scores = cross_val_score(model, X_sub, y_sub, cv=tscv, scoring="accuracy")

        results.append({
            "fraction": frac,
            "sample_size": n,
            "acc_mean": float(scores.mean()),
            "acc_std": float(scores.std()),
        })

    if len(results) >= 2:
        low = results[0]["acc_mean"]
        high = results[-1]["acc_mean"]
        convergence = "Converged" if abs(high - low) < 0.05 else "Not Converged"
    else:
        convergence = "Insufficient Data"

    return {
        "results": results,
        "convergence_status": convergence,
    }


# ── Report Generation ─────────────────────────────────────────────────


def generate_report(lc, cv, fi, ss, symbol, timeframe, n_samples):
    """Generate markdown report with all analysis results."""
    report = f"""# 過擬合分析報告

生成時間: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

## 概要
| 項目 | 數值 |
|------|------|
| 交易對 | {symbol} |
| 時間框架 | {timeframe} |
| 樣本數 | {n_samples} |
| CV平均精度 | {cv['acc_mean']:.2%} ± {cv['acc_std']:.2%} |
| CV平均F1 | {cv['f1_mean']:.4f} ± {cv['f1_std']:.4f} |
| 最大訓練-CV差距 | {lc['max_gap']:.2%} |
| 特徵穩定比例 | {fi['stable_count']}/{fi['total_features']} ({fi['stability_ratio']:.0%}) |
| 收斂狀態 | {ss['convergence_status']} |

## 1. 學習曲線分析

訓練精度和驗證精度隨訓練集大小的變化：

| 訓練集大小 | 訓練精度 | 驗證精度 | 差距 |
|------------|----------|----------|------|
"""
    for i in range(len(lc['train_sizes'])):
        train_pct = lc['train_mean'][i]
        val_pct = lc['val_mean'][i]
        gap = lc['train_mean'][i] - lc['val_mean'][i]
        report += f"| {lc['train_sizes'][i]:.0f} | {train_pct:.2%} | {val_pct:.2%} | {gap:.2%} |\n"

    report += f"""
**結論**: 訓練-驗證差距為 {lc['final_gap']:.2%}，最佳驗證精度為 {lc['best_val_score']:.2%}。
"""
    if lc['final_gap'] > 0.15:
        report += "⚠️ 存在明顯過擬合迹象（差距 > 15%）。\n"
    elif lc['final_gap'] > 0.10:
        report += "⚠️ 輕度過擬合迹象（差距 > 10%），建議增加正則化。\n"
    else:
        report += "✅ 模型泛化能力良好（差距 < 10%）。\n"

    report += f"""
## 2. 時間序列交叉驗證（{cv['n_splits']}-Fold）

| Fold | 訓練集大小 | 測試集大小 | 精度 | F1 |
|------|------------|------------|------|-----|
"""
    for fold in cv['fold_details']:
        report += f"| {fold['fold']} | {fold['train_size']} | {fold['test_size']} | {fold['acc']:.2%} | {fold['f1']:.4f} |\n"

    report += f"""
**結論**: 各折精度標準差為 {cv['acc_std']:.2%}。
"""
    if cv['acc_std'] > 0.05:
        report += "⚠️ 高方差（標準差 > 5%），模型對時間段敏感。\n"
    else:
        report += "✅ 模型在不同時間段表現穩定。\n"

    report += """
## 3. 特徵重要性穩定性

| 排名 | 特徵 | 平均重要性 | 標準差 | 變異係數 |
|------|------|------------|--------|----------|
"""
    for i, feat in enumerate(fi['features']):
        report += f"| {i+1} | {feat['feature']} | {feat['mean_importance']:.4f} | {feat['std_importance']:.4f} | {feat['cv']:.2f} |\n"

    report += f"""
**結論**: {fi['stable_count']}/{fi['total_features']} 個特徵在多次訓練中保持穩定（CV < 0.5）。
"""
    if fi['stability_ratio'] < 0.5:
        report += "⚠️ 特徵重要性波動較大，可能存在過擬合。\n"
    else:
        report += "✅ 特徵重要性較穩定。\n"

    report += """
## 4. 樣本量分析

| 訓練比例 | 樣本數 | 平均精度 | 標準差 |
|----------|--------|----------|--------|
"""
    for r in ss['results']:
        report += f"| {r['fraction']:.0%} | {r['sample_size']} | {r['acc_mean']:.2%} | {r['acc_std']:.2%} |\n"

    report += f"""
**結論**: 收斂狀態 — {ss['convergence_status']}。
"""
    if ss['convergence_status'] == "Not Converged":
        report += "⚠️ 增加數據可能繼續提升性能，建議收集更多歷史數據。\n"
    else:
        report += "✅ 模型性能已接近收斂。\n"

    report += """
## 5. 綜合建議

"""
    issues = []
    if lc['final_gap'] > 0.10:
        issues.append("過擬合風險")
    if cv['acc_std'] > 0.05:
        issues.append("時間敏感性")
    if fi['stability_ratio'] < 0.5:
        issues.append("特徵不穩定")
    if ss['convergence_status'] == "Not Converged":
        issues.append("數據不足")

    if not issues:
        report += "- ✅ 模型整體表現良好，無明顯過擬合迹象。\n"
        report += "- 建議定期重新訓練以適應市場變化。\n"
    else:
        report += f"- ⚠️ 存在以下問題：{', '.join(issues)}\n"
        report += "- 建議措施：\n"
        if "過擬合風險" in issues:
            report += "  - 增加正則化參數（降低 max_depth 或增加 min_samples_leaf）\n"
            report += "  - 減少特徵數量（使用特徵選擇）\n"
        if "時間敏感性" in issues:
            report += "  - 使用更多交叉驗證折數\n"
            report += "  - 考慮使用滾動窗口訓練\n"
        if "特徵不穩定" in issues:
            report += "  - 移除不穩定特徵\n"
            report += "  - 使用特徵選擇方法\n"
        if "數據不足" in issues:
            report += "  - 收集更多歷史數據\n"
            report += "  - 考慮使用更短的時間框架\n"

    return report


# ── Main ──────────────────────────────────────────────────────────────


def main():
    symbol = "BTC/USDT"
    timeframe = "1h"
    limit = 8000

    print(f"Loading data for {symbol} {timeframe}...")
    X, y, df = prepare_data(symbol, timeframe, limit)
    print(f"  {len(X)} samples, {len(X.columns)} features")

    print("Running learning curve analysis...")
    lc = learning_curve_analysis(X, y)
    print(f"  Max gap: {lc['max_gap']:.2%}, Best val: {lc['best_val_score']:.2%}")

    print("Running cross-validation...")
    cv = cross_validation_analysis(X, y)
    print(f"  CV accuracy: {cv['acc_mean']:.2%} ± {cv['acc_std']:.2%}")

    print("Running feature importance stability...")
    fi = feature_importance_stability(X, y)
    print(f"  Stable features: {fi['stable_count']}/{fi['total_features']}")

    print("Running sample size analysis...")
    ss = sample_size_analysis(X, y)
    print(f"  Convergence: {ss['convergence_status']}")

    report = generate_report(lc, cv, fi, ss, symbol, timeframe, len(X))
    report_path = "overfitting_analysis.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
