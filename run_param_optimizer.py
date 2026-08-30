#!/usr/bin/env python
"""Parameter optimization script to reduce overfitting via regularization.

Tests different hyperparameter combinations for RandomForest and XGBoost,
ranking by train-validation gap (overfitting) and validation accuracy.
"""

import sys
import time
import warnings
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

from data_cache import load_or_fetch


# ── Feature Engineering ──────────────────────────────────────────


def compute_local_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators."""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    return pd.DataFrame({
        "SMA_20": ta.trend.sma_indicator(close, window=20),
        "SMA_50": ta.trend.sma_indicator(close, window=50),
        "RSI": ta.momentum.rsi(close, window=14),
        "MACD": ta.trend.MACD(close).macd(),
        "ATR": ta.volatility.average_true_range(high, low, close, window=14),
        "MFI": ta.volume.money_flow_index(high, low, close, volume, window=14),
        "OBV": ta.volume.on_balance_volume(close, volume),
    }, index=df.index)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix with momentum indicators."""
    indicators = compute_local_indicators(df)
    indicators["close"] = df["close"]

    rsi_norm = (indicators["RSI"] - 50) / 50
    macd_norm = np.tanh(indicators["MACD"] / indicators["MACD"].std())
    sma20_norm = (indicators["close"] - indicators["SMA_20"]) / indicators["SMA_20"]
    sma50_norm = (indicators["close"] - indicators["SMA_50"]) / indicators["SMA_50"]

    indicators["momentum_score"] = (
        0.3 * rsi_norm + 0.1 * macd_norm + 0.4 * sma20_norm + 0.2 * sma50_norm
    ).clip(-1, 1)
    indicators["momentum_delta"] = indicators["momentum_score"].diff()
    indicators["momentum_acceleration"] = indicators["momentum_delta"].diff()

    return indicators


def generate_labels(df: pd.DataFrame, n_bars: int = 5) -> pd.Series:
    """Binary labels: 1 = price goes up in next n_bars."""
    future = df["close"].shift(-n_bars)
    ret = ((future - df["close"]) / df["close"]) * 100
    labels = (ret > 0).astype(float)
    labels[ret.isna()] = np.nan
    return labels


# ── Parameter Grids ──────────────────────────────────────────────

RF_PARAM_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7, 10, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 5, 10],
}

XGB_PARAM_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
}


# ── Evaluation ───────────────────────────────────────────────────


def evaluate_rf(params, X_train, X_test, y_train, y_test, scaler):
    """Evaluate RandomForest with given parameters."""
    model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model.fit(X_train_s, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train_s))
    val_scores = cross_val_score(model, X_train_s, y_train, cv=5, scoring="accuracy")
    val_acc = val_scores.mean()

    y_pred = model.predict(X_test_s)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, model.predict_proba(X_test_s)[:, 1])
    except ValueError:
        auc = 0.0

    return {
        "train_acc": train_acc,
        "val_acc": val_acc,
        "gap": train_acc - val_acc,
        "f1": f1,
        "auc": auc,
        "params": params,
    }


def evaluate_xgb(params, X_train, X_test, y_train, y_test, scaler):
    """Evaluate XGBoost with given parameters."""
    model = XGBClassifier(
        **params, random_state=42, eval_metric="logloss", verbosity=0
    )
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model.fit(X_train_s, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train_s))
    val_scores = cross_val_score(model, X_train_s, y_train, cv=5, scoring="accuracy")
    val_acc = val_scores.mean()

    y_pred = model.predict(X_test_s)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, model.predict_proba(X_test_s)[:, 1])
    except ValueError:
        auc = 0.0

    return {
        "train_acc": train_acc,
        "val_acc": val_acc,
        "gap": train_acc - val_acc,
        "f1": f1,
        "auc": auc,
        "params": params,
    }


# ── Grid Search ──────────────────────────────────────────────────


def run_grid_search(param_grid, X_train, X_test, y_train, y_test, evaluate_fn):
    """Run exhaustive grid search over parameter combinations."""
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(product(*values))
    results = []

    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        scaler = StandardScaler()
        try:
            result = evaluate_fn(params, X_train, X_test, y_train, y_test, scaler)
            results.append(result)
        except Exception as e:
            print(f"  [{i}/{len(combos)}] Error: {e}")
            continue

        if i % 20 == 0 or i == len(combos):
            print(f"  [{i}/{len(combos)}] Last gap={result['gap']:.3f}, val={result['val_acc']:.3f}")

    return results


# ── Report Generation ────────────────────────────────────────────


def generate_report(rf_results, xgb_results, rf_time, xgb_time, meta):
    """Generate markdown optimization report."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    rf_sorted = sorted(rf_results, key=lambda x: (x["gap"], -x["val_acc"]))
    xgb_sorted = sorted(xgb_results, key=lambda x: (x["gap"], -x["val_acc"]))

    # Build RF table
    rf_rows = []
    for rank, r in enumerate(rf_sorted[:10], 1):
        p = r["params"]
        depth = p["max_depth"] if p["max_depth"] is not None else "None"
        rf_rows.append(
            f"| {rank} | {p['n_estimators']} | {depth} | {p['min_samples_split']} "
            f"| {p['min_samples_leaf']} | {r['train_acc']*100:.1f} | {r['val_acc']*100:.1f} "
            f"| {r['gap']*100:.1f} | {r['f1']:.2f} |"
        )

    # Build XGB table
    xgb_rows = []
    for rank, r in enumerate(xgb_sorted[:10], 1):
        p = r["params"]
        xgb_rows.append(
            f"| {rank} | {p['n_estimators']} | {p['max_depth']} | {p['learning_rate']} "
            f"| {p['subsample']} | {p['colsample_bytree']} | {r['train_acc']*100:.1f} "
            f"| {r['val_acc']*100:.1f} | {r['gap']*100:.1f} | {r['f1']:.2f} |"
        )

    # Pick best model (lowest gap, then highest val_acc)
    all_results = [(r, "RF") for r in rf_sorted] + [(r, "XGB") for r in xgb_sorted]
    all_results.sort(key=lambda x: (x[0]["gap"], -x[0]["val_acc"]))
    best_result, best_model = all_results[0]
    best_params = best_result["params"]

    if best_model == "RF":
        depth_str = best_params["max_depth"] if best_params["max_depth"] is not None else "None"
        best_params_str = (
            f"n_estimators={best_params['n_estimators']}, "
            f"max_depth={depth_str}, "
            f"min_samples_split={best_params['min_samples_split']}, "
            f"min_samples_leaf={best_params['min_samples_leaf']}"
        )
        usage_code = f"""from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators={best_params['n_estimators']},
    max_depth={best_params['max_depth'] if best_params['max_depth'] is not None else 'None'},
    min_samples_split={best_params['min_samples_split']},
    min_samples_leaf={best_params['min_samples_leaf']},
    random_state=42
)"""
    else:
        best_params_str = (
            f"n_estimators={best_params['n_estimators']}, "
            f"max_depth={best_params['max_depth']}, "
            f"learning_rate={best_params['learning_rate']}, "
            f"subsample={best_params['subsample']}, "
            f"colsample_bytree={best_params['colsample_bytree']}"
        )
        usage_code = f"""from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators={best_params['n_estimators']},
    max_depth={best_params['max_depth']},
    learning_rate={best_params['learning_rate']},
    subsample={best_params['subsample']},
    colsample_bytree={best_params['colsample_bytree']},
    random_state=42,
    eval_metric="logloss"
)"""

    report = f"""# 參數優化報告

生成時間: {now}

## 測試參數組合
| 模型 | 參數組合數 | 測試時間 |
|------|------------|----------|
| RandomForest | {len(rf_results)} | {rf_time:.0f}秒 |
| XGBoost | {len(xgb_results)} | {xgb_time:.0f}秒 |

## 測試參數範圍
| 模型 | 參數 | 範圍 |
|------|------|------|
| RF | n_estimators | {RF_PARAM_GRID['n_estimators']} |
| RF | max_depth | {RF_PARAM_GRID['max_depth']} |
| RF | min_samples_split | {RF_PARAM_GRID['min_samples_split']} |
| RF | min_samples_leaf | {RF_PARAM_GRID['min_samples_leaf']} |
| XGB | n_estimators | {XGB_PARAM_GRID['n_estimators']} |
| XGB | max_depth | {XGB_PARAM_GRID['max_depth']} |
| XGB | learning_rate | {XGB_PARAM_GRID['learning_rate']} |
| XGB | subsample | {XGB_PARAM_GRID['subsample']} |
| XGB | colsample_bytree | {XGB_PARAM_GRID['colsample_bytree']} |

## 當前問題
| 項目 | 數值 |
|------|------|
| 原始訓練精度 | ~97% |
| 原始驗證精度 | ~52% |
| 原始差距 | ~45% |

## RandomForest 最佳參數 (Top 10)
| 排名 | n_est | max_depth | min_split | min_leaf | 訓練% | 驗證% | 差距% | F1 |
|------|-------|-----------|-----------|----------|-------|-------|-------|-----|
{chr(10).join(rf_rows)}

## XGBoost 最佳參數 (Top 10)
| 排名 | n_est | max_depth | lr | subsample | colsample | 訓練% | 驗證% | 差距% | F1 |
|------|-------|-----------|-----|-----------|-----------|-------|-------|-------|-----|
{chr(10).join(xgb_rows)}

## 最終推薦
| 項目 | 數值 |
|------|------|
| 推薦模型 | {best_model} |
| 最佳參數 | {best_params_str} |
| 預期驗證精度 | {best_result['val_acc']*100:.1f}% |
| 預期差距 | {best_result['gap']*100:.1f}% |
| F1 Score | {best_result['f1']:.4f} |
| AUC | {best_result['auc']:.4f} |

## 使用方式
```python
{usage_code}

# Fit on scaled training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
model.fit(X_train_scaled, y_train)
```

## 分析結論
1. **過擬合問題**: 原始模型差距 {meta['original_gap']*100:.1f}% 表明嚴重過擬合
2. **最佳策略**: {best_model} 在降低差距方面表現最佳
3. **預期改善**: 驗證精度可達 {best_result['val_acc']*100:.1f}%，差距降至 {best_result['gap']*100:.1f}%
4. **建議**: 使用推薦參數重新訓練模型，並監控驗證集表現
"""
    return report


# ── Main ─────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("參數優化器 — 測試正則化參數以減少過擬合")
    print("=" * 60)

    # 1. Load data
    print("\n[1/5] 載入快取數據...")
    df = load_or_fetch("BTC/USDT", "1h", limit=8000)
    print(f"  數據筆數: {len(df)} ({df.index[0]} ~ {df.index[-1]})")

    # 2. Build features
    print("\n[2/5] 建構特徵...")
    features = build_features(df)
    labels = generate_labels(df)

    valid = features.dropna().index.intersection(labels.dropna().index)
    X = features.loc[valid]
    y = labels.loc[valid]
    print(f"  有效樣本: {len(X)}")

    # Train/test split (80/20, no shuffle for time series)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    print(f"  訓練集: {len(X_train)}, 測試集: {len(X_test)}")

    feature_columns = [c for c in X.columns if c != "close"]
    X_train = X_train[feature_columns]
    X_test = X_test[feature_columns]

    # Baseline gap
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    baseline_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    baseline_rf.fit(X_train_s, y_train)
    baseline_train = accuracy_score(y_train, baseline_rf.predict(X_train_s))
    baseline_val_scores = cross_val_score(baseline_rf, X_train_s, y_train, cv=5, scoring="accuracy")
    baseline_val = baseline_val_scores.mean()
    baseline_gap = baseline_train - baseline_val
    print(f"\n  基準: 訓練={baseline_train:.3f}, 驗證={baseline_val:.3f}, 差距={baseline_gap:.3f}")

    # 3. RF grid search
    print(f"\n[3/5] RandomForest 網格搜索 ({len(list(product(*RF_PARAM_GRID.values())))} 組合)...")
    rf_combos = list(product(*RF_PARAM_GRID.values()))
    print(f"  預計測試 {len(rf_combos)} 種參數組合")
    t0 = time.time()
    rf_results = run_grid_search(RF_PARAM_GRID, X_train, X_test, y_train, y_test, evaluate_rf)
    rf_time = time.time() - t0
    print(f"  完成: {len(rf_results)} 結果, 耗時 {rf_time:.0f}秒")

    # 4. XGB grid search
    print(f"\n[4/5] XGBoost 網格搜索 ({len(list(product(*XGB_PARAM_GRID.values())))} 組合)...")
    xgb_combos = list(product(*XGB_PARAM_GRID.values()))
    print(f"  預計測試 {len(xgb_combos)} 種參數組合")
    t0 = time.time()
    xgb_results = run_grid_search(XGB_PARAM_GRID, X_train, X_test, y_train, y_test, evaluate_xgb)
    xgb_time = time.time() - t0
    print(f"  完成: {len(xgb_results)} 結果, 耗時 {xgb_time:.0f}秒")

    # 5. Generate report
    print("\n[5/5] 生成報告...")
    meta = {"original_gap": baseline_gap}
    report = generate_report(rf_results, xgb_results, rf_time, xgb_time, meta)
    report_path = "param_optimization_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  報告已保存: {report_path}")

    # Summary
    all_results = [(r, "RF") for r in rf_results] + [(r, "XGB") for r in xgb_results]
    all_results.sort(key=lambda x: (x[0]["gap"], -x[0]["val_acc"]))
    best_result, best_model = all_results[0]

    print(f"\n{'=' * 60}")
    print("最佳參數")
    print(f"{'=' * 60}")
    print(f"  模型: {best_model}")
    print(f"  參數: {best_result['params']}")
    print(f"  訓練精度: {best_result['train_acc']*100:.1f}%")
    print(f"  驗證精度: {best_result['val_acc']*100:.1f}%")
    print(f"  差距: {best_result['gap']*100:.1f}%")
    print(f"  F1: {best_result['f1']:.4f}")
    print(f"  AUC: {best_result['auc']:.4f}")
    print(f"{'=' * 60}")
    print(f"\nStatus: DONE")
    print(f"Files: run_param_optimizer.py, param_optimization_report.md")


if __name__ == "__main__":
    main()
