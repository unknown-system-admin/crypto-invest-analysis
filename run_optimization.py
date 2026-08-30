#!/usr/bin/env python3
"""Comprehensive optimization script for crypto investment analysis."""
import sys
sys.path.insert(0, ".")

import warnings
warnings.filterwarnings("ignore")

import ccxt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from itertools import product
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

from feature_engine.builder import build_feature_matrix
from feature_engine.momentum import momentum_score as compute_momentum_score
from feature_engine.indicators import compute_all_indicators


def fetch_data():
    """Fetch BTC/USDT daily data from OKX."""
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


def prepare_data(df):
    """Build feature matrix and labels."""
    features, labels = build_feature_matrix(df, n_bars=5)
    ml_features = features.copy()
    ml_features["binary_label"] = labels
    ml_features = ml_features.dropna()
    feature_columns = [col for col in ml_features.columns if col != "binary_label"]
    return ml_features, feature_columns


def get_feature_importance(X, y):
    """Train a basic model and return feature importances."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    importances = dict(zip(X.columns, model.feature_importances_))
    return importances, model


def optimization_a_redundant_features(ml_features, feature_columns):
    """A. Remove Redundant Features - remove highly correlated features."""
    print("\n" + "=" * 60)
    print("OPTIMIZATION A: Remove Redundant Features")
    print("=" * 60)

    X = ml_features[feature_columns]
    y = ml_features["binary_label"]

    # Get feature importances
    importances, baseline_model = get_feature_importance(X, y)
    baseline_pred = baseline_model.predict(X)
    baseline_acc = accuracy_score(y, baseline_pred)
    print(f"  Baseline accuracy (all features): {baseline_acc:.4f}")
    print(f"  Total features: {len(feature_columns)}")

    # Compute correlation matrix
    corr_matrix = X.corr().abs()

    # Find highly correlated pairs
    threshold = 0.8
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    correlated_pairs = []
    for col in upper.columns:
        for idx in upper.index:
            if upper.loc[idx, col] > threshold:
                correlated_pairs.append((idx, col, upper.loc[idx, col]))

    print(f"\n  Correlated pairs (|r| > {threshold}): {len(correlated_pairs)}")
    for f1, f2, corr in correlated_pairs:
        print(f"    {f1} <-> {f2}: {corr:.3f}")

    # Remove redundant features (keep the one with higher importance)
    features_to_remove = set()
    for f1, f2, corr in correlated_pairs:
        imp1 = importances.get(f1, 0)
        imp2 = importances.get(f2, 0)
        if imp1 < imp2:
            features_to_remove.add(f1)
        else:
            features_to_remove.add(f2)

    reduced_features = [f for f in feature_columns if f not in features_to_remove]
    print(f"\n  Features to remove: {sorted(features_to_remove)}")
    print(f"  Remaining features: {len(reduced_features)}")

    # Test reduced feature set
    X_reduced = ml_features[reduced_features]
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    reduced_acc = accuracy_score(y_test, model.predict(X_test))

    print(f"\n  Reduced model accuracy: {reduced_acc:.4f}")
    print(f"  Accuracy change: {reduced_acc - baseline_acc:+.4f}")
    print(f"  Features removed: {len(features_to_remove)}")

    return {
        "removed_features": sorted(features_to_remove),
        "reduced_features": reduced_features,
        "baseline_accuracy": baseline_acc,
        "reduced_accuracy": reduced_acc,
        "accuracy_change": reduced_acc - baseline_acc,
        "correlated_pairs": correlated_pairs,
    }


def optimization_b_regularization(ml_features, feature_columns):
    """B. Regularize Model - test different regularization parameters."""
    print("\n" + "=" * 60)
    print("OPTIMIZATION B: Regularize Model")
    print("=" * 60)

    X = ml_features[feature_columns]
    y = ml_features["binary_label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Parameter grid
    param_grid = {
        "max_depth": [3, 5, 7, 10, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 5, 10],
    }

    best_score = 0
    best_params = {}
    results = []

    print("  Testing parameter combinations...")
    total = len(param_grid["max_depth"]) * len(param_grid["min_samples_split"]) * len(param_grid["min_samples_leaf"])

    for i, (depth, split, leaf) in enumerate(product(
        param_grid["max_depth"],
        param_grid["min_samples_split"],
        param_grid["min_samples_leaf"],
    )):
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=depth,
            min_samples_split=split,
            min_samples_leaf=leaf,
            random_state=42,
        )

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        mean_cv = cv_scores.mean()

        model.fit(X_train, y_train)
        test_acc = accuracy_score(y_test, model.predict(X_test))

        results.append({
            "max_depth": depth,
            "min_samples_split": split,
            "min_samples_leaf": leaf,
            "cv_accuracy": round(mean_cv, 4),
            "test_accuracy": round(test_acc, 4),
        })

        if mean_cv > best_score:
            best_score = mean_cv
            best_params = {"max_depth": depth, "min_samples_split": split, "min_samples_leaf": leaf}

    # Train best model
    best_model = RandomForestClassifier(n_estimators=100, random_state=42, **best_params)
    best_model.fit(X_train, y_train)
    best_test_acc = accuracy_score(y_test, best_model.predict(X_test))

    # Baseline (no regularization)
    baseline = RandomForestClassifier(n_estimators=100, random_state=42)
    baseline.fit(X_train, y_train)
    baseline_acc = accuracy_score(y_test, baseline.predict(X_test))

    print(f"\n  Best parameters: {best_params}")
    print(f"  Best CV accuracy: {best_score:.4f}")
    print(f"  Best test accuracy: {best_test_acc:.4f}")
    print(f"  Baseline test accuracy: {baseline_acc:.4f}")
    print(f"  Improvement: {best_test_acc - baseline_acc:+.4f}")

    # Top 5 results
    results_sorted = sorted(results, key=lambda x: x["cv_accuracy"], reverse=True)
    print("\n  Top 5 parameter combinations:")
    for r in results_sorted[:5]:
        print(f"    depth={r['max_depth']}, split={r['min_samples_split']}, leaf={r['min_samples_leaf']}: "
              f"CV={r['cv_accuracy']:.4f}, Test={r['test_accuracy']:.4f}")

    return {
        "best_params": best_params,
        "best_cv_accuracy": best_score,
        "best_test_accuracy": best_test_acc,
        "baseline_accuracy": baseline_acc,
        "all_results": results_sorted[:10],
    }


def optimization_c_feature_engineering(ml_features, feature_columns):
    """C. Feature Engineering - create and test combination features."""
    print("\n" + "=" * 60)
    print("OPTIMIZATION C: Feature Engineering")
    print("=" * 60)

    X = ml_features[feature_columns]
    y = ml_features["binary_label"]

    # Baseline
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    baseline_model = RandomForestClassifier(n_estimators=100, random_state=42)
    baseline_model.fit(X_train, y_train)
    baseline_acc = accuracy_score(y_test, baseline_model.predict(X_test))
    print(f"  Baseline accuracy: {baseline_acc:.4f}")

    # Create engineered features
    engineered = ml_features.copy()

    # RSI_MACD = RSI * MACD_histogram
    if "RSI" in engineered.columns and "MACD_histogram" in engineered.columns:
        engineered["RSI_MACD"] = engineered["RSI"] * engineered["MACD_histogram"]
        print("  Created: RSI_MACD")

    # Price_SMA_ratio = close / SMA_20
    if "SMA_20" in engineered.columns:
        engineered["Price_SMA_ratio"] = engineered["close"] / engineered["SMA_20"]
        print("  Created: Price_SMA_ratio")
    elif "close" in ml_features.columns:
        # Try to find close in original features
        pass

    # BB_position = (close - BB_lower) / (BB_upper - BB_lower)
    if all(c in engineered.columns for c in ["BB_lower", "BB_upper"]):
        if "close" in engineered.columns:
            engineered["BB_position"] = (engineered["close"] - engineered["BB_lower"]) / (
                engineered["BB_upper"] - engineered["BB_lower"]
            )
            print("  Created: BB_position")
        elif "SMA_20" in engineered.columns:
            # Use SMA_20 as proxy for close
            engineered["BB_position"] = (engineered["SMA_20"] - engineered["BB_lower"]) / (
                engineered["BB_upper"] - engineered["BB_lower"]
            )
            print("  Created: BB_position (using SMA_20 as close proxy)")

    # Momentum_RSI = momentum_score * RSI
    if "momentum_score" in engineered.columns and "RSI" in engineered.columns:
        engineered["Momentum_RSI"] = engineered["momentum_score"] * engineered["RSI"]
        print("  Created: Momentum_RSI")

    # Drop NaN from new features
    engineered = engineered.dropna()
    eng_features = [col for col in engineered.columns if col != "binary_label"]

    X_eng = engineered[eng_features]
    y_eng = engineered["binary_label"]

    X_train_eng, X_test_eng, y_train_eng, y_test_eng = train_test_split(
        X_eng, y_eng, test_size=0.2, random_state=42
    )

    eng_model = RandomForestClassifier(n_estimators=100, random_state=42)
    eng_model.fit(X_train_eng, y_train_eng)
    eng_acc = accuracy_score(y_test_eng, eng_model.predict(X_test_eng))

    print(f"\n  Engineered model accuracy: {eng_acc:.4f}")
    print(f"  Accuracy change: {eng_acc - baseline_acc:+.4f}")
    print(f"  New features added: {len(eng_features) - len(feature_columns)}")

    # Feature importances of new features
    new_features = [f for f in eng_features if f not in feature_columns]
    if new_features:
        importances = dict(zip(eng_features, eng_model.feature_importances_))
        print("\n  New feature importances:")
        for f in new_features:
            print(f"    {f}: {importances.get(f, 0):.4f}")

    return {
        "baseline_accuracy": baseline_acc,
        "engineered_accuracy": eng_acc,
        "accuracy_change": eng_acc - baseline_acc,
        "new_features": new_features,
        "feature_columns": eng_features,
    }


def optimization_d_time_series_cv(ml_features, feature_columns):
    """D. Time Series Cross-Validation."""
    print("\n" + "=" * 60)
    print("OPTIMIZATION D: Time Series Cross-Validation")
    print("=" * 60)

    X = ml_features[feature_columns].values
    y = ml_features["binary_label"].values

    # Standard k-fold
    standard_scores = cross_val_score(
        RandomForestClassifier(n_estimators=100, random_state=42),
        X, y, cv=5, scoring="accuracy"
    )

    # Time series split
    tscv = TimeSeriesSplit(n_splits=5)
    ts_scores = cross_val_score(
        RandomForestClassifier(n_estimators=100, random_state=42),
        X, y, cv=tscv, scoring="accuracy"
    )

    print(f"  Standard 5-Fold CV:")
    print(f"    Mean accuracy: {standard_scores.mean():.4f} (+/- {standard_scores.std():.4f})")
    print(f"    Fold scores: {[round(s, 4) for s in standard_scores]}")

    print(f"\n  Time Series 5-Fold CV:")
    print(f"    Mean accuracy: {ts_scores.mean():.4f} (+/- {ts_scores.std():.4f})")
    print(f"    Fold scores: {[round(s, 4) for s in ts_scores]}")

    # Train/test split simulation
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    temporal_acc = accuracy_score(y_test, model.predict(X_test))

    print(f"\n  Temporal split (80/20):")
    print(f"    Test accuracy: {temporal_acc:.4f}")
    print(f"    Train size: {split_idx}, Test size: {len(X) - split_idx}")

    print(f"\n  Key insight: Time series CV is more realistic for financial data")
    print(f"  Standard CV may overestimate by {standard_scores.mean() - ts_scores.mean():+.4f}")

    return {
        "standard_cv_mean": round(standard_scores.mean(), 4),
        "standard_cv_std": round(standard_scores.std(), 4),
        "standard_cv_folds": [round(s, 4) for s in standard_scores],
        "timeseries_cv_mean": round(ts_scores.mean(), 4),
        "timeseries_cv_std": round(ts_scores.std(), 4),
        "timeseries_cv_folds": [round(s, 4) for s in ts_scores],
        "temporal_split_accuracy": round(temporal_acc, 4),
    }


def optimization_e_momentum_weights(ml_features, feature_columns, df):
    """E. Optimize Momentum Formula weights."""
    print("\n" + "=" * 60)
    print("OPTIMIZATION E: Optimize Momentum Formula")
    print("=" * 60)

    indicators = compute_all_indicators(df)
    indicators["close"] = df["close"]

    # Current weights
    current_weights = (0.3, 0.3, 0.2, 0.2)

    # Test weight combinations
    weight_options = [0.1, 0.2, 0.3, 0.4, 0.5]
    best_score = 0
    best_weights = current_weights
    results = []

    print("  Testing momentum weight combinations...")

    for w_rsi, w_macd, w_sma20, w_sma50 in product(weight_options, repeat=4):
        total = w_rsi + w_macd + w_sma20 + w_sma50
        if abs(total - 1.0) > 0.01:
            continue  # Skip if weights don't sum to 1

        # Compute normalized indicators
        rsi_norm = (indicators["RSI"] - 50) / 50
        macd_norm = np.tanh(indicators["MACD_histogram"] / indicators["MACD_histogram"].std())
        sma20_norm = (indicators["close"] - indicators["SMA_20"]) / indicators["SMA_20"]
        sma50_norm = (indicators["close"] - indicators["SMA_50"]) / indicators["SMA_50"]

        score = (
            w_rsi * rsi_norm +
            w_macd * macd_norm +
            w_sma20 * sma20_norm +
            w_sma50 * sma50_norm
        ).clip(-1, 1)

        # Build feature matrix with this momentum
        temp_features = indicators.copy()
        temp_features["momentum_score"] = score
        temp_features["momentum_delta"] = score.diff()
        temp_features["momentum_acceleration"] = temp_features["momentum_delta"].diff()

        from feature_engine.labels import binary_label
        labels = binary_label(df, n_bars=5)

        ml_data = temp_features.drop(columns=["close"], errors="ignore")
        ml_data["binary_label"] = labels
        ml_data = ml_data.dropna()

        if len(ml_data) < 50:
            continue

        feature_cols = [c for c in ml_data.columns if c != "binary_label"]
        X = ml_data[feature_cols]
        y = ml_data["binary_label"]

        tscv = TimeSeriesSplit(n_splits=5)
        scores = cross_val_score(
            RandomForestClassifier(n_estimators=100, random_state=42),
            X.values, y.values, cv=tscv, scoring="accuracy"
        )

        mean_score = scores.mean()
        results.append({
            "weights": (w_rsi, w_macd, w_sma20, w_sma50),
            "cv_accuracy": round(mean_score, 4),
        })

        if mean_score > best_score:
            best_score = mean_score
            best_weights = (w_rsi, w_macd, w_sma20, w_sma50)

    results_sorted = sorted(results, key=lambda x: x["cv_accuracy"], reverse=True)

    print(f"\n  Current weights: RSI={current_weights[0]}, MACD={current_weights[1]}, "
          f"SMA20={current_weights[2]}, SMA50={current_weights[3]}")

    # Evaluate current weights
    rsi_norm = (indicators["RSI"] - 50) / 50
    macd_norm = np.tanh(indicators["MACD_histogram"] / indicators["MACD_histogram"].std())
    sma20_norm = (indicators["close"] - indicators["SMA_20"]) / indicators["SMA_20"]
    sma50_norm = (indicators["close"] - indicators["SMA_50"]) / indicators["SMA_50"]

    current_score = (
        current_weights[0] * rsi_norm +
        current_weights[1] * macd_norm +
        current_weights[2] * sma20_norm +
        current_weights[3] * sma50_norm
    ).clip(-1, 1)

    temp_features = indicators.copy()
    temp_features["momentum_score"] = current_score
    temp_features["momentum_delta"] = current_score.diff()
    temp_features["momentum_acceleration"] = temp_features["momentum_delta"].diff()

    from feature_engine.labels import binary_label
    labels = binary_label(df, n_bars=5)

    ml_data = temp_features.drop(columns=["close"], errors="ignore")
    ml_data["binary_label"] = labels
    ml_data = ml_data.dropna()

    feature_cols = [c for c in ml_data.columns if c != "binary_label"]
    X = ml_data[feature_cols]
    y = ml_data["binary_label"]

    tscv = TimeSeriesSplit(n_splits=5)
    current_cv = cross_val_score(
        RandomForestClassifier(n_estimators=100, random_state=42),
        X.values, y.values, cv=tscv, scoring="accuracy"
    ).mean()

    print(f"  Current weights CV accuracy: {current_cv:.4f}")

    print(f"\n  Best weights: RSI={best_weights[0]}, MACD={best_weights[1]}, "
          f"SMA20={best_weights[2]}, SMA50={best_weights[3]}")
    print(f"  Best CV accuracy: {best_score:.4f}")
    print(f"  Improvement: {best_score - current_cv:+.4f}")

    print("\n  Top 5 weight combinations:")
    for r in results_sorted[:5]:
        w = r["weights"]
        print(f"    RSI={w[0]}, MACD={w[1]}, SMA20={w[2]}, SMA50={w[3]}: {r['cv_accuracy']:.4f}")

    return {
        "current_weights": current_weights,
        "current_cv_accuracy": round(current_cv, 4),
        "best_weights": best_weights,
        "best_cv_accuracy": best_score,
        "improvement": round(best_score - current_cv, 4),
        "top_5": results_sorted[:5],
    }


def generate_report(results_a, results_b, results_c, results_d, results_e):
    """Generate optimization report in markdown."""
    report = """# Crypto Investment Analysis - Optimization Report

## Summary

This report documents the results of 5 optimization techniques applied to the crypto investment analysis model.

---

## A. Remove Redundant Features

### Method
- Removed features with |correlation| > 0.8
- Kept the feature with higher importance from each correlated pair

### Results
- **Baseline accuracy:** {baseline_acc:.4f}
- **Reduced model accuracy:** {reduced_acc:.4f}
- **Accuracy change:** {accuracy_change:+.4f}
- **Features removed:** {n_removed}

### Correlated Pairs Found
{correlated_pairs}

### Features Removed
{removed_features}

---

## B. Regularize Model

### Method
- Tested combinations of max_depth, min_samples_split, min_samples_leaf
- Used 5-fold cross-validation to select best parameters

### Results
- **Best parameters:** {best_params}
- **Best CV accuracy:** {best_cv:.4f}
- **Best test accuracy:** {best_test:.4f}
- **Baseline test accuracy:** {baseline_test:.4f}

### Top 5 Configurations
{top5_configs}

---

## C. Feature Engineering

### Method
- Created combination features: RSI_MACD, Price_SMA_ratio, BB_position, Momentum_RSI
- Tested if engineered features improve model performance

### Results
- **Baseline accuracy:** {eng_baseline:.4f}
- **Engineered model accuracy:** {eng_result:.4f}
- **Accuracy change:** {eng_change:+.4f}
- **New features added:** {n_new_features}

### New Features
{new_features_list}

---

## D. Time Series Cross-Validation

### Method
- Compared standard 5-fold CV with TimeSeriesSplit (5 splits)
- Also tested temporal 80/20 train/test split

### Results
- **Standard 5-Fold CV:** {std_cv_mean:.4f} (+/- {std_cv_std:.4f})
- **Time Series 5-Fold CV:** {ts_cv_mean:.4f} (+/- {ts_cv_std:.4f})
- **Temporal Split Accuracy:** {temporal_acc:.4f}

### Fold Scores
| Fold | Standard | Time Series |
|------|----------|-------------|
{fold_table}

### Key Insight
Standard CV may overestimate performance by {cv_difference:+.4f} compared to time series CV.

---

## E. Optimize Momentum Formula

### Method
- Tested different weight combinations for momentum_score
- Weights: RSI, MACD, SMA20, SMA50 (must sum to 1.0)

### Results
- **Current weights:** RSI={curr_w0}, MACD={curr_w1}, SMA20={curr_w2}, SMA50={curr_w3}
- **Current CV accuracy:** {curr_cv:.4f}
- **Best weights:** RSI={best_w0}, MACD={best_w1}, SMA20={best_w2}, SMA50={best_w3}
- **Best CV accuracy:** {momentum_best_cv:.4f}
- **Improvement:** {momentum_improvement:+.4f}

### Top 5 Weight Combinations
{top5_weights}

---

## Recommendations

1. **Feature Selection:** Remove {n_removed_text} redundant features to simplify the model
2. **Regularization:** Use max_depth={best_depth} to prevent overfitting
3. **Feature Engineering:** {eng_recommendation}
4. **Validation:** Always use time series cross-validation for financial data
5. **Momentum:** Consider updating weights to {best_w0}/{best_w1}/{best_w2}/{best_w3}
""".format(
        baseline_acc=results_a["baseline_accuracy"],
        reduced_acc=results_a["reduced_accuracy"],
        accuracy_change=results_a["accuracy_change"],
        n_removed=len(results_a["removed_features"]),
        correlated_pairs="\n".join(
            [f"- {f1} <-> {f2}: r={corr:.3f}" for f1, f2, corr in results_a["correlated_pairs"]]
        ) or "- None found",
        removed_features=", ".join(results_a["removed_features"]) or "None",
        best_params=str(results_b["best_params"]),
        best_cv=results_b["best_cv_accuracy"],
        best_test=results_b["best_test_accuracy"],
        baseline_test=results_b["baseline_accuracy"],
        n_removed_text=len(results_a["removed_features"]),
        top5_configs="\n".join(
            [f"| {r['max_depth']} | {r['min_samples_split']} | {r['min_samples_leaf']} | {r['cv_accuracy']:.4f} | {r['test_accuracy']:.4f} |"
             for r in results_b["all_results"][:5]]
        ),
        eng_baseline=results_c["baseline_accuracy"],
        eng_result=results_c["engineered_accuracy"],
        eng_change=results_c["accuracy_change"],
        n_new_features=len(results_c["new_features"]),
        new_features_list=", ".join(results_c["new_features"]) or "None",
        std_cv_mean=results_d["standard_cv_mean"],
        std_cv_std=results_d["standard_cv_std"],
        ts_cv_mean=results_d["timeseries_cv_mean"],
        ts_cv_std=results_d["timeseries_cv_std"],
        temporal_acc=results_d["temporal_split_accuracy"],
        fold_table="\n".join(
            [f"| {i+1} | {results_d['standard_cv_folds'][i]:.4f} | {results_d['timeseries_cv_folds'][i]:.4f} |"
             for i in range(len(results_d['standard_cv_folds']))]
        ),
        cv_difference=results_d["standard_cv_mean"] - results_d["timeseries_cv_mean"],
        curr_w0=results_e["current_weights"][0],
        curr_w1=results_e["current_weights"][1],
        curr_w2=results_e["current_weights"][2],
        curr_w3=results_e["current_weights"][3],
        curr_cv=results_e["current_cv_accuracy"],
        best_w0=results_e["best_weights"][0],
        best_w1=results_e["best_weights"][1],
        best_w2=results_e["best_weights"][2],
        best_w3=results_e["best_weights"][3],
        momentum_best_cv=results_e["best_cv_accuracy"],
        momentum_improvement=results_e["improvement"],
        top5_weights="\n".join(
            [f"| RSI={r['weights'][0]} | MACD={r['weights'][1]} | SMA20={r['weights'][2]} | SMA50={r['weights'][3]} | {r['cv_accuracy']:.4f} |"
             for r in results_e["top_5"]]
        ),
        best_depth=results_b["best_params"]["max_depth"],
        eng_recommendation="Add engineered features" if results_c["accuracy_change"] > 0 else "Engineered features did not improve performance",
    )

    return report


def main():
    print("=" * 60)
    print("CRYPTO INVESTMENT ANALYSIS - OPTIMIZATION SUITE")
    print("=" * 60)

    # Fetch data
    df = fetch_data()

    # Prepare data
    ml_features, feature_columns = prepare_data(df)

    # ML feature columns (exclude close for model training)
    ml_feature_columns = [c for c in feature_columns if c != "close"]

    # Run all optimizations
    results_a = optimization_a_redundant_features(ml_features, ml_feature_columns)
    results_b = optimization_b_regularization(ml_features, ml_feature_columns)
    results_c = optimization_c_feature_engineering(ml_features, ml_feature_columns)
    results_d = optimization_d_time_series_cv(ml_features, ml_feature_columns)
    results_e = optimization_e_momentum_weights(ml_features, ml_feature_columns, df)

    # Generate report
    print("\n" + "=" * 60)
    print("GENERATING REPORT")
    print("=" * 60)

    report = generate_report(results_a, results_b, results_c, results_d, results_e)

    report_path = "optimization_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nReport saved to: {report_path}")
    print("=" * 60)
    print("ALL OPTIMIZATIONS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
