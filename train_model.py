#!/usr/bin/env python
"""Model training script with customizable parameters and save/load support."""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import ta
import ccxt
from sklearn.ensemble import RandomForestClassifier
from data_cache import load_or_fetch, update_cache
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
import joblib


# ── Local Feature Engine ──────────────────────────────────────────────


def compute_local_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators locally (not imported from modules)."""
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
    """Momentum score with optimized weights, range [-1, 1]."""
    rsi_norm = (df["RSI"] - 50) / 50
    macd_norm = np.tanh(df["MACD"] / df["MACD"].std())
    sma20_norm = (df["close"] - df["SMA_20"]) / df["SMA_20"]
    sma50_norm = (df["close"] - df["SMA_50"]) / df["SMA_50"]

    score = 0.3 * rsi_norm + 0.1 * macd_norm + 0.4 * sma20_norm + 0.2 * sma50_norm
    return score.clip(-1, 1)


def build_features(df: pd.DataFrame, use_momentum: bool, use_volume: bool,
                   use_volatility: bool) -> pd.DataFrame:
    """Build feature matrix from OHLCV data."""
    indicators = compute_local_indicators(df)

    if use_momentum:
        indicators["close"] = df["close"]
        indicators["momentum_score"] = compute_momentum_score(indicators)
        indicators["momentum_delta"] = indicators["momentum_score"].diff()

    if not use_volume:
        indicators = indicators.drop(columns=["MFI", "OBV"], errors="ignore")

    if not use_volatility:
        indicators = indicators.drop(columns=["ATR"], errors="ignore")

    return indicators


def generate_labels(df: pd.DataFrame, n_bars: int = 5) -> pd.Series:
    """Binary labels: 1 = price goes up in next n_bars, 0 = down."""
    future = df["close"].shift(-n_bars)
    ret = ((future - df["close"]) / df["close"]) * 100
    labels = (ret > 0).astype(float)
    labels[ret.isna()] = np.nan
    return labels


# ── Data Fetching ─────────────────────────────────────────────────────


def fetch_ohlcv(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    """Fetch OHLCV data from OKX."""
    exchange = ccxt.okx({
        "apiKey": os.getenv("OKX_API_KEY"),
        "secret": os.getenv("OKX_API_SECRET"),
        "password": os.getenv("OKX_API_PASSPHRASE"),
        "enableRateLimit": True,
    })
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df


# ── Model Factory ─────────────────────────────────────────────────────


def create_model(args):
    """Create model based on CLI arguments."""
    if args.model == "rf":
        return RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            random_state=42,
            n_jobs=-1,
        )
    elif args.model == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
        )
    elif args.model == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            random_state=42,
            verbose=-1,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")


# ── Training ──────────────────────────────────────────────────────────


def train_and_evaluate(args):
    """Main training pipeline."""
    symbol = "BTC/USDT"

    print(f"Loading {symbol} {args.timeframe} data ({args.limit} candles)...")
    if args.update_cache:
        df = update_cache(symbol, args.timeframe, args.limit)
        print(f"  Cache updated: {len(df)} candles: {df.index[0]} -> {df.index[-1]}")
    else:
        df = load_or_fetch(symbol, args.timeframe, args.limit)
        print(f"  Got {len(df)} candles: {df.index[0]} -> {df.index[-1]}")

    print("Building features...")
    features = build_features(df, args.use_momentum, args.use_volume, args.use_volatility)
    labels = generate_labels(df)

    valid = features.dropna().index.intersection(labels.dropna().index)
    X = features.loc[valid]
    y = labels.loc[valid]
    print(f"  Samples: {len(X)} (after dropping NaN)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 - args.train_ratio, shuffle=False
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if args.load_model:
        print(f"Loading model from {args.load_model}...")
        model = joblib.load(args.load_model)
    else:
        model = create_model(args)
        print(f"Training {args.model.upper()} (n_est={args.n_estimators}, depth={args.max_depth})...")
        model.fit(X_train_s, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train_s))
    test_acc = accuracy_score(y_test, model.predict(X_test_s))
    y_pred = model.predict(X_test_s)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, model.predict_proba(X_test_s)[:, 1])
    except ValueError:
        auc = 0.0

    print(f"\n{'='*40}")
    print(f"Train accuracy: {train_acc:.2%}")
    print(f"Test accuracy:  {test_acc:.2%}")
    print(f"F1 Score:       {f1:.4f}")
    print(f"AUC:            {auc:.4f}")
    print(f"{'='*40}")
    print("\nClassification Report (test set):")
    print(classification_report(y_test, y_pred, target_names=["Down", "Up"]))

    if args.save_model:
        joblib.dump(model, args.save_model)
        scaler_path = os.path.join(os.path.dirname(args.save_model) or ".", "scaler.pkl")
        joblib.dump(scaler, scaler_path)
        print(f"Model saved to {args.save_model}")
        print(f"Scaler saved to {scaler_path}")

    if args.output_report:
        generate_report(args, df, X_train, X_test, train_acc, test_acc, f1, auc, model, X.columns.tolist())

    return model, scaler, X.columns.tolist()


# ── Report ────────────────────────────────────────────────────────────


def generate_report(args, df, X_train, X_test, train_acc, test_acc, f1, auc,
                    model, feature_names):
    """Generate training report in markdown."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    n_bars = len(df)

    # Feature importance
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        fi = sorted(zip(feature_names, importances), key=lambda x: -x[1])
    else:
        fi = [(n, 0.0) for n in feature_names]

    tf_map = {"1h": "1h", "4h": "4h", "1d": "1d"}
    tf_display = tf_map.get(args.timeframe, args.timeframe)

    fi_rows = "\n".join(f"| {i+1} | {name} | {imp:.4f} |" for i, (name, imp) in enumerate(fi))

    report = f"""# 模型訓練報告

生成時間: {now}

## 訓練參數
| 參數 | 數值 |
|------|------|
| 模型類型 | {args.model.upper()} |
| n_estimators | {args.n_estimators} |
| max_depth | {args.max_depth} |
| min_samples_split | {args.min_samples_split} |
| min_samples_leaf | {args.min_samples_leaf} |
| learning_rate | {args.learning_rate} |
| subsample | {args.subsample} |
| colsample_bytree | {args.colsample_bytree} |

## 資料摘要
| 項目 | 數值 |
|------|------|
| 時間框架 | {tf_display} |
| 總筆數 | {n_bars} |
| 訓練集 | {len(X_train)} |
| 測試集 | {len(X_test)} |

## 訓練結果
| 指標 | 數值 |
|------|------|
| 訓練精度 | {train_acc:.2%} |
| 測試精度 | {test_acc:.2%} |
| F1 Score | {f1:.4f} |
| AUC | {auc:.4f} |

## 特徵重要性
| 排名 | 特徵 | 重要性 |
|------|------|--------|
{fi_rows}

## 使用方式
```python
from joblib import load

model = load('{args.save_model or "model.pkl"}')
scaler = load('scaler.pkl')
```
"""
    report_path = "model_training_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report saved to {report_path}")


# ── CLI ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Train crypto prediction model with customizable parameters.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:

1. Train RandomForest with default parameters:
   python train_model.py --model rf

2. Train XGBoost with custom parameters:
   python train_model.py --model xgb --n-estimators 200 --max-depth 5 --learning-rate 0.05

3. Train and save model:
   python train_model.py --model rf --save-model model_rf.pkl --output-report

4. Load and evaluate existing model:
   python train_model.py --load-model model_rf.pkl
""",
    )

    model_group = parser.add_argument_group("Model Options")
    model_group.add_argument("--model", choices=["rf", "xgb", "lgbm"], default="rf",
                             help="Model type (default: rf)")
    model_group.add_argument("--n-estimators", type=int, default=100,
                             help="Number of trees (default: 100)")
    model_group.add_argument("--max-depth", type=int, default=7,
                             help="Max tree depth (default: 7)")
    model_group.add_argument("--min-samples-split", type=int, default=5,
                             help="Min samples to split (default: 5)")
    model_group.add_argument("--min-samples-leaf", type=int, default=2,
                             help="Min samples in leaf (default: 2)")
    model_group.add_argument("--learning-rate", type=float, default=0.1,
                             help="Learning rate for XGB/LGBM (default: 0.1)")
    model_group.add_argument("--subsample", type=float, default=0.8,
                             help="Subsample ratio (default: 0.8)")
    model_group.add_argument("--colsample-bytree", type=float, default=0.8,
                             help="Column sample ratio (default: 0.8)")

    data_group = parser.add_argument_group("Data Options")
    data_group.add_argument("--timeframe", choices=["1h", "4h", "1d"], default="1h",
                            help="Timeframe (default: 1h)")
    data_group.add_argument("--limit", type=int, default=8000,
                            help="Number of candles (default: 8000)")
    data_group.add_argument("--train-ratio", type=float, default=0.8,
                            help="Train/test split ratio (default: 0.8)")

    feature_group = parser.add_argument_group("Feature Options")
    feature_group.add_argument("--use-momentum", action="store_true", default=True,
                               help="Use momentum features (default: True)")
    feature_group.add_argument("--use-volume", action="store_true", default=True,
                               help="Use volume features (default: True)")
    feature_group.add_argument("--use-volatility", action="store_true", default=True,
                               help="Use volatility features (default: True)")

    cache_group = parser.add_argument_group("Cache Options")
    cache_group.add_argument("--update-cache", action="store_true",
                             help="Force update cache with fresh data from OKX")

    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument("--save-model", type=str, default=None,
                              help="Save model to file")
    output_group.add_argument("--load-model", type=str, default=None,
                              help="Load existing model")
    output_group.add_argument("--output-report", action="store_true",
                              help="Generate training report")

    args = parser.parse_args()
    train_and_evaluate(args)


if __name__ == "__main__":
    main()
