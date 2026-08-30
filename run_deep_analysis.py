#!/usr/bin/env python3
"""Deep analysis: timeframe, threshold search, cross-validation, correlation, model comparison."""
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
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from feature_engine.builder import build_feature_matrix
from feature_engine.momentum import momentum_score, momentum_delta
from backtest_engine.engine import BacktestEngine
from backtest_engine.rule_strategy import MomentumRuleStrategy
from backtest_engine.model_strategy import ModelStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fetch_ohlcv(timeframe: str, days: int = 365) -> pd.DataFrame:
    """Fetch OHLCV from OKX for a given timeframe."""
    exchange = ccxt.okx()
    since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

    limit_map = {"1h": 1000, "4h": 1000, "1d": 365}
    limit = limit_map.get(timeframe, 365)

    all_candles = []
    cursor = since
    while True:
        candles = exchange.fetch_ohlcv("BTC/USDT", timeframe, since=cursor, limit=limit)
        if not candles:
            break
        all_candles.extend(candles)
        cursor = candles[-1][0] + 1
        if len(candles) < limit:
            break

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df[~df.index.duplicated(keep="first")]
    return df


def run_backtest(features: pd.DataFrame, timeframe: str = "1d",
                 buy_threshold: float = 0.3, sell_threshold: float = -0.3) -> dict:
    """Run a backtest with MomentumRuleStrategy."""
    strategy = MomentumRuleStrategy(buy_threshold=buy_threshold, sell_threshold=sell_threshold)
    engine = BacktestEngine(strategy=strategy, timeframe=timeframe)
    result = engine.run(features)
    return {
        "total_trades": result.total_trades,
        "total_return_pct": result.total_return_pct,
        "max_drawdown_pct": result.max_drawdown_pct,
        "sharpe_ratio": result.sharpe_ratio,
        "win_rate": result.win_rate,
    }


def run_ml_backtest(features: pd.DataFrame, labels: pd.Series, timeframe: str = "1d",
                    buy_threshold: float = 0.5, sell_threshold: float = -0.5) -> dict:
    """Run a backtest with ModelStrategy using RandomForest."""
    ml_df = features.copy()
    ml_df["binary_label"] = labels
    ml_df = ml_df.dropna()
    feature_cols = [c for c in ml_df.columns if c != "binary_label"]
    if len(ml_df) < 50:
        return {"total_trades": 0, "total_return_pct": 0, "max_drawdown_pct": 0,
                "sharpe_ratio": 0, "win_rate": 0}

    X = ml_df[feature_cols].values
    y = ml_df["binary_label"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    strategy = ModelStrategy(model, feature_cols, confidence_threshold=0.5)
    engine = BacktestEngine(strategy=strategy, timeframe=timeframe)
    result = engine.run(features.dropna())
    return {
        "total_trades": result.total_trades,
        "total_return_pct": result.total_return_pct,
        "max_drawdown_pct": result.max_drawdown_pct,
        "sharpe_ratio": result.sharpe_ratio,
        "win_rate": result.win_rate,
    }


# ---------------------------------------------------------------------------
# Analysis A: Timeframe Analysis
# ---------------------------------------------------------------------------

def analysis_timeframe() -> str:
    print("\n" + "=" * 60)
    print("A. TIMEFRAME ANALYSIS")
    print("=" * 60)

    timeframes = {"1h": 365, "4h": 365, "1d": 365}
    results = {}

    for tf, days in timeframes.items():
        print(f"\n  Fetching {tf} data...")
        try:
            df = fetch_ohlcv(tf, days=days)
            print(f"    Rows: {len(df)}, Range: {df.index[0].date()} - {df.index[-1].date()}")
        except Exception as e:
            print(f"    Error fetching {tf}: {e}")
            continue

        try:
            features, labels = build_feature_matrix(df, n_bars=5)
            print(f"    Features: {features.shape[1]}, Samples: {features.shape[0]}")
        except Exception as e:
            print(f"    Error building features for {tf}: {e}")
            continue

        bt = run_backtest(features, timeframe=tf)
        print(f"    Backtest: return={bt['total_return_pct']:.1f}%, sharpe={bt['sharpe_ratio']:.2f}, "
              f"trades={bt['total_trades']}, win_rate={bt['win_rate']:.1f}%")

        ml_bt = run_ml_backtest(features, labels, timeframe=tf)
        print(f"    ML Backtest: return={ml_bt['total_return_pct']:.1f}%, sharpe={ml_bt['sharpe_ratio']:.2f}")

        results[tf] = {"rule": bt, "ml": ml_bt, "samples": features.shape[0]}

    # Build report
    lines = ["## A. Timeframe Analysis\n"]
    lines.append("Comparing backtest results across 1h, 4h, and 1d timeframes using BTC/USDT.\n")
    lines.append("| Timeframe | Samples | Rule Return% | Rule Sharpe | Rule WinRate | ML Return% | ML Sharpe |")
    lines.append("|-----------|---------|--------------|-------------|--------------|------------|-----------|")
    for tf, r in results.items():
        lines.append(
            f"| {tf} | {r['samples']} | {r['rule']['total_return_pct']:.1f} | "
            f"{r['rule']['sharpe_ratio']:.2f} | {r['rule']['win_rate']:.1f}% | "
            f"{r['ml']['total_return_pct']:.1f} | {r['ml']['sharpe_ratio']:.2f} |"
        )

    if results:
        best_rule = max(results.items(), key=lambda x: x[1]["rule"]["sharpe_ratio"])
        best_ml = max(results.items(), key=lambda x: x[1]["ml"]["sharpe_ratio"])
        lines.append(f"\n**Best rule-based timeframe:** {best_rule[0]} (Sharpe {best_rule[1]['rule']['sharpe_ratio']:.2f})")
        lines.append(f"**Best ML timeframe:** {best_ml[0]} (Sharpe {best_ml[1]['ml']['sharpe_ratio']:.2f})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analysis B: Fine-grained Threshold Search
# ---------------------------------------------------------------------------

def analysis_threshold_search() -> str:
    print("\n" + "=" * 60)
    print("B. THRESHOLD SEARCH (81 combinations)")
    print("=" * 60)

    df = fetch_ohlcv("1d", days=365)
    features, labels = build_feature_matrix(df, n_bars=5)

    buy_thresholds = np.arange(0.1, 0.55, 0.05)
    sell_thresholds = np.arange(-0.5, -0.05, 0.05)

    results = []
    total = len(buy_thresholds) * len(sell_thresholds)
    count = 0

    for buy_t, sell_t in product(buy_thresholds, sell_thresholds):
        count += 1
        if count % 10 == 0:
            print(f"  Testing {count}/{total}...")
        bt = run_backtest(features, timeframe="1d",
                          buy_threshold=round(buy_t, 2), sell_threshold=round(sell_t, 2))
        results.append({
            "buy": round(buy_t, 2),
            "sell": round(sell_t, 2),
            **bt,
        })

    best = max(results, key=lambda x: x["sharpe_ratio"])
    print(f"\n  Best: buy={best['buy']}, sell={best['sell']}")
    print(f"    Return: {best['total_return_pct']:.1f}%, Sharpe: {best['sharpe_ratio']:.2f}")
    print(f"    Trades: {best['total_trades']}, WinRate: {best['win_rate']:.1f}%")

    # Top 5
    top5 = sorted(results, key=lambda x: x["sharpe_ratio"], reverse=True)[:5]

    lines = ["## B. Threshold Search\n"]
    lines.append(f"Tested {len(buy_thresholds)} x {len(sell_thresholds)} = {total} combinations.\n")
    lines.append("### Top 5 by Sharpe Ratio\n")
    lines.append("| Buy | Sell | Return% | Sharpe | Trades | WinRate | MaxDD% |")
    lines.append("|-----|------|---------|--------|--------|---------|--------|")
    for r in top5:
        lines.append(
            f"| {r['buy']:.2f} | {r['sell']:.2f} | {r['total_return_pct']:.1f} | "
            f"{r['sharpe_ratio']:.2f} | {r['total_trades']} | {r['win_rate']:.1f}% | "
            f"{r['max_drawdown_pct']:.1f}% |"
        )

    lines.append(f"\n**Best combination:** buy={best['buy']}, sell={best['sell']}")
    lines.append(f"- Return: {best['total_return_pct']:.1f}%")
    lines.append(f"- Sharpe: {best['sharpe_ratio']:.2f}")
    lines.append(f"- Win Rate: {best['win_rate']:.1f}%")
    lines.append(f"- Max Drawdown: {best['max_drawdown_pct']:.1f}%")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analysis C: Cross-Validation
# ---------------------------------------------------------------------------

def analysis_cross_validation() -> str:
    print("\n" + "=" * 60)
    print("C. CROSS-VALIDATION (5-fold)")
    print("=" * 60)

    df = fetch_ohlcv("1d", days=365)
    features, labels = build_feature_matrix(df, n_bars=5)

    ml_df = features.copy()
    ml_df["binary_label"] = labels
    ml_df = ml_df.dropna()
    feature_cols = [c for c in ml_df.columns if c != "binary_label"]

    X = ml_df[feature_cols].values
    y = ml_df["binary_label"].values

    print(f"  Samples: {len(X)}, Features: {len(feature_cols)}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    scores = cross_val_score(model, X_scaled, y, cv=5, scoring="accuracy")

    print(f"  Fold accuracies: {[f'{s:.4f}' for s in scores]}")
    print(f"  Mean: {scores.mean():.4f} (+/- {scores.std():.4f})")

    # Also report train/test split for comparison
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    test_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"  Holdout test accuracy: {test_acc:.4f}")

    lines = ["## C. Cross-Validation\n"]
    lines.append(f"5-fold cross-validation on {len(X)} samples with {len(feature_cols)} features.\n")
    lines.append(f"| Fold | Accuracy |")
    lines.append(f"|------|----------|")
    for i, s in enumerate(scores):
        lines.append(f"| {i+1} | {s:.4f} |")
    lines.append(f"| **Mean** | **{scores.mean():.4f}** |")
    lines.append(f"| **Std** | **{scores.std():.4f}** |\n")
    lines.append(f"**Holdout test accuracy:** {test_acc:.4f}")
    lines.append(f"\n**Interpretation:** CV mean {scores.mean():.4f} ± {scores.std():.4f} provides a more robust "
                 f"estimate than the single holdout split ({test_acc:.4f}).")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analysis D: Feature Correlation Matrix
# ---------------------------------------------------------------------------

def analysis_correlation() -> str:
    print("\n" + "=" * 60)
    print("D. FEATURE CORRELATION MATRIX")
    print("=" * 60)

    df = fetch_ohlcv("1d", days=365)
    features, labels = build_feature_matrix(df, n_bars=5)

    # Drop close if present
    features = features.drop(columns=["close"], errors="ignore")
    features = features.dropna()

    print(f"  Features: {features.shape[1]}, Samples: {features.shape[0]}")

    corr = features.corr()

    # Find highly correlated pairs
    high_corr = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if abs(r) > 0.8:
                high_corr.append((cols[i], cols[j], round(r, 3)))

    high_corr.sort(key=lambda x: abs(x[2]), reverse=True)

    print(f"  Highly correlated pairs (|r| > 0.8): {len(high_corr)}")
    for a, b, r in high_corr:
        print(f"    {a} <-> {b}: {r}")

    # Suggest features to remove (keep the one with lower average correlation)
    remove_set = set()
    for a, b, r in high_corr:
        avg_corr_a = corr[a].drop(a).abs().mean()
        avg_corr_b = corr[b].drop(b).abs().mean()
        if avg_corr_a > avg_corr_b:
            remove_set.add(a)
        else:
            remove_set.add(b)

    lines = ["## D. Feature Correlation Matrix\n"]
    lines.append(f"Computed correlation matrix for {features.shape[1]} features.\n")

    if high_corr:
        lines.append("### Highly Correlated Pairs (|r| > 0.8)\n")
        lines.append("| Feature A | Feature B | Correlation |")
        lines.append("|-----------|-----------|-------------|")
        for a, b, r in high_corr:
            lines.append(f"| {a} | {b} | {r} |")

        lines.append(f"\n### Suggested Features to Remove\n")
        lines.append(f"Based on average correlation with other features, consider removing:\n")
        for f in sorted(remove_set):
            lines.append(f"- `{f}`")
    else:
        lines.append("No highly correlated pairs (|r| > 0.8) found.\n")

    lines.append(f"\n### Full Correlation Summary\n")
    lines.append(f"- Total features: {features.shape[1]}")
    lines.append(f"- Highly correlated pairs: {len(high_corr)}")
    lines.append(f"- Suggested removals: {len(remove_set)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analysis E: Model Comparison
# ---------------------------------------------------------------------------

def analysis_model_comparison() -> str:
    print("\n" + "=" * 60)
    print("E. MODEL COMPARISON")
    print("=" * 60)

    df = fetch_ohlcv("1d", days=365)
    features, labels = build_feature_matrix(df, n_bars=5)

    ml_df = features.copy()
    ml_df["binary_label"] = labels
    ml_df = ml_df.dropna()
    feature_cols = [c for c in ml_df.columns if c != "binary_label"]

    X = ml_df[feature_cols].values
    y = ml_df["binary_label"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"  Train: {len(X_train)}, Test: {len(X_test)}, Features: {len(feature_cols)}")

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss", verbosity=0),
        "LightGBM": LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Feature importance
        if hasattr(model, "feature_importances_"):
            imp = dict(zip(feature_cols, model.feature_importances_))
            top3 = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:3]
        else:
            top3 = []

        results[name] = {"accuracy": acc, "top_features": top3}
        print(f"  {name}: accuracy={acc:.4f}, top features={[f[0] for f in top3]}")

    best_name = max(results, key=lambda x: results[x]["accuracy"])
    print(f"\n  Best model: {best_name} ({results[best_name]['accuracy']:.4f})")

    lines = ["## E. Model Comparison\n"]
    lines.append("Comparing Random Forest, XGBoost, and LightGBM on the same train/test split.\n")
    lines.append("| Model | Accuracy | Top Feature 1 | Top Feature 2 | Top Feature 3 |")
    lines.append("|-------|----------|---------------|---------------|---------------|")
    for name, r in results.items():
        top = r["top_features"]
        t1 = f"{top[0][0]} ({top[0][1]:.3f})" if len(top) > 0 else "N/A"
        t2 = f"{top[1][0]} ({top[1][1]:.3f})" if len(top) > 1 else "N/A"
        t3 = f"{top[2][0]} ({top[2][1]:.3f})" if len(top) > 2 else "N/A"
        marker = " **BEST**" if name == best_name else ""
        lines.append(f"| {name}{marker} | {r['accuracy']:.4f} | {t1} | {t2} | {t3} |")

    lines.append(f"\n**Best model:** {best_name} with accuracy {results[best_name]['accuracy']:.4f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("DEEP ANALYSIS")
    print("=" * 60)

    sections = []

    try:
        sections.append(analysis_timeframe())
    except Exception as e:
        print(f"ERROR in Timeframe Analysis: {e}")
        sections.append(f"## A. Timeframe Analysis\n\n**Error:** {e}")

    try:
        sections.append(analysis_threshold_search())
    except Exception as e:
        print(f"ERROR in Threshold Search: {e}")
        sections.append(f"## B. Threshold Search\n\n**Error:** {e}")

    try:
        sections.append(analysis_cross_validation())
    except Exception as e:
        print(f"ERROR in Cross-Validation: {e}")
        sections.append(f"## C. Cross-Validation\n\n**Error:** {e}")

    try:
        sections.append(analysis_correlation())
    except Exception as e:
        print(f"ERROR in Correlation Matrix: {e}")
        sections.append(f"## D. Feature Correlation Matrix\n\n**Error:** {e}")

    try:
        sections.append(analysis_model_comparison())
    except Exception as e:
        print(f"ERROR in Model Comparison: {e}")
        sections.append(f"## E. Model Comparison\n\n**Error:** {e}")

    # Assemble report
    report_lines = [
        "# Deep Analysis Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Symbol:** BTC/USDT",
        "",
        "---",
        "",
    ]
    report_lines.extend(sections)

    report = "\n\n".join(report_lines)

    report_path = "deep_analysis_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n{'=' * 60}")
    print(f"Report saved to: {report_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
