#!/usr/bin/env python3
"""Final multi-timeframe strategy optimizer: tests rule-based and ML strategies
across 1d, 4h, and 1h timeframes to find the best overall performer."""
import sys
sys.path.insert(0, ".")

import warnings
warnings.filterwarnings("ignore")

import ccxt
import numpy as np
import pandas as pd
import ta
import time
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from backtest_engine.engine import BacktestEngine
from backtest_engine.rule_strategy import MomentumRuleStrategy
from backtest_engine.strategy import Strategy, Signal


# ── Data Fetching ──────────────────────────────────────────────

def fetch_ohlcv(symbol, timeframe, limit):
    """Fetch OHLCV data from OKX with pagination."""
    print(f"  Fetching {limit} candles of {symbol} {timeframe}...")
    exchange = ccxt.okx({"enableRateLimit": True})
    all_data = []

    # Calculate how far back to start
    tf_minutes = {"1h": 60, "4h": 240, "1d": 1440}.get(timeframe, 60)
    start_offset = (limit * tf_minutes + 1440) * 60 * 1000  # extra day buffer
    since = int((datetime.now() - timedelta(milliseconds=start_offset)).timestamp() * 1000)

    while len(all_data) < limit:
        try:
            batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=300)
        except Exception as e:
            print(f"    Error: {e}, retrying...")
            time.sleep(2)
            continue
        if not batch:
            break
        all_data.extend(batch)
        since = batch[-1][0] + 1
        if len(batch) < 300:
            break
        time.sleep(0.2)

    all_data = all_data[:limit]
    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df[~df.index.duplicated(keep="first")]
    print(f"    Period: {df.index[0]} ~ {df.index[-1]} ({len(df)} rows)")
    return df


# ── Local Feature Engine ───────────────────────────────────────

def compute_indicators(df):
    """Compute local technical indicators."""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    sma_20 = ta.trend.sma_indicator(close, window=20)
    sma_50 = ta.trend.sma_indicator(close, window=50)
    rsi = ta.momentum.rsi(close, window=14)
    macd_obj = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    macd_line = macd_obj.macd()
    atr = ta.volatility.average_true_range(high, low, close, window=14)
    mfi = ta.volume.money_flow_index(high, low, close, volume, window=14)
    obv = ta.volume.on_balance_volume(close, volume)

    return pd.DataFrame({
        "SMA_20": sma_20,
        "SMA_50": sma_50,
        "RSI": rsi,
        "MACD": macd_line,
        "ATR": atr,
        "MFI": mfi,
        "OBV": obv,
    }, index=df.index)


def build_features(df):
    """Build feature matrix with momentum score and labels."""
    indicators = compute_indicators(df)
    indicators["close"] = df["close"]

    # Momentum score: weighted composite
    # RSI=0.3, MACD=0.1, SMA20=0.4, SMA50=0.2
    indicators["momentum_score"] = (
        (indicators["RSI"] - 50) / 50 * 0.3
        + indicators["MACD"] / indicators["close"].abs() * 100 * 0.1
        + (indicators["close"] / indicators["SMA_20"] - 1) * 0.4
        + (indicators["close"] / indicators["SMA_50"] - 1) * 0.2
    )
    indicators["momentum_delta"] = indicators["momentum_score"].diff()
    indicators["momentum_acceleration"] = indicators["momentum_delta"].diff()

    # Binary label: price up in next 5 bars
    n_bars = 5
    future_return = df["close"].shift(-n_bars) / df["close"] - 1
    labels = (future_return > 0).astype(int)

    valid_idx = indicators.dropna().index.intersection(labels.dropna().index)
    features = indicators.loc[valid_idx]
    labels = labels.loc[valid_idx]
    return features, labels


# ── Backtest ───────────────────────────────────────────────────

def run_backtest(strategy, features_df, timeframe="1h"):
    """Run backtest and return result."""
    engine = BacktestEngine(
        strategy=strategy,
        initial_capital=10000.0,
        fee_rate=0.001,
        slippage=0.0005,
        max_position_pct=25.0,
        max_daily_trades=50,
        symbol="BTC/USDT",
        max_drawdown_stop=30.0,
        timeframe=timeframe,
    )
    return engine.run(features_df)


def extract_metrics(result):
    """Extract key metrics from backtest result."""
    return {
        "total_return": result.total_return_pct,
        "sharpe": result.sharpe_ratio,
        "max_drawdown": result.max_drawdown_pct,
        "num_trades": result.total_trades,
        "win_rate": result.win_rate,
    }


# ── Rule-Based Strategies ──────────────────────────────────────

RULE_CONFIGS = [
    ("Aggressive 0.01", 0.01, -0.01),
    ("Aggressive 0.02", 0.02, -0.02),
    ("Aggressive 0.03", 0.03, -0.03),
    ("Aggressive 0.05", 0.05, -0.05),
    ("Aggressive 0.08", 0.08, -0.07),
]


def test_rule_strategies(features, timeframe):
    """Test all rule-based strategy variants."""
    results = []
    for name, buy_th, sell_th in RULE_CONFIGS:
        strategy = MomentumRuleStrategy(buy_threshold=buy_th, sell_threshold=sell_th)
        bt_result = run_backtest(strategy, features, timeframe)
        m = extract_metrics(bt_result)
        m["name"] = name
        m["type"] = "Rule"
        m["buy_th"] = buy_th
        m["sell_th"] = sell_th
        results.append(m)
    return results


# ── ML Strategies ──────────────────────────────────────────────

class MLThresholdStrategy(Strategy):
    """ML model with configurable probability threshold."""
    def __init__(self, model, feature_columns, scaler, buy_threshold=0.5):
        self.model = model
        self.feature_columns = feature_columns
        self.scaler = scaler
        self.buy_threshold = buy_threshold

    def evaluate(self, features) -> Signal:
        try:
            raw = np.array([[features.get(col, 0) for col in self.feature_columns]])
            X = self.scaler.transform(raw)
            proba = self.model.predict_proba(X)[0]
            buy_prob = proba[1]
            if buy_prob >= self.buy_threshold:
                return Signal("偏多", buy_prob, "ml")
            elif buy_prob <= (1 - self.buy_threshold):
                return Signal("偏空", 1 - buy_prob, "ml")
            else:
                return Signal("中立", 0.5, "ml")
        except Exception:
            return Signal("中立", 0.0, "ml")


def test_ml_models(features, labels, feature_columns, split_idx, timeframe):
    """Train RF and XGB, test with multiple thresholds."""
    X = features[feature_columns].values
    y = labels.values
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "RF": RandomForestClassifier(n_estimators=100, max_depth=7,
                                     min_samples_split=5, random_state=42),
        "XGB": XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             random_state=42, eval_metric="logloss", verbosity=0),
    }

    test_features = features.iloc[split_idx:].copy()
    thresholds = [0.3, 0.4, 0.5]
    results = []

    for model_name, model in models.items():
        model.fit(X_train_scaled, y_train)
        test_acc = model.score(X_test_scaled, y_test)

        for th in thresholds:
            strategy = MLThresholdStrategy(model, feature_columns, scaler, buy_threshold=th)
            bt_result = run_backtest(strategy, test_features, timeframe)
            m = extract_metrics(bt_result)
            m["name"] = f"{model_name} (th={th})"
            m["type"] = "ML"
            m["test_accuracy"] = round(test_acc, 4)
            results.append(m)

    return results


# ── Report Generation ──────────────────────────────────────────

def generate_report(tf_results, all_results):
    """Generate the final markdown report."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    best = max(all_results, key=lambda x: x["total_return"])

    # Determine best type
    if best["type"] == "ML":
        best_type = "ML"
    else:
        best_type = "規則"

    # Build per-timeframe tables
    tf_tables = ""
    for tf in ["1d", "4h", "1h"]:
        tf_label = {"1d": "1d (日線)", "4h": "4h (4小時)", "1h": "1h (小時)"}[tf]
        rows = sorted(tf_results.get(tf, []), key=lambda x: x["total_return"], reverse=True)
        tf_tables += f"### {tf_label}\n"
        tf_tables += "| 策略 | Buy | Sell | 報酬% | Sharpe | 交易次數 |\n"
        tf_tables += "|------|-----|------|-------|--------|----------|\n"
        for r in rows:
            if r["type"] == "Rule":
                tf_tables += (f"| {r['name']} | {r['buy_th']} | {r['sell_th']} | "
                              f"{r['total_return']:.1f}% | {r['sharpe']:.2f} | "
                              f"{r['num_trades']} |\n")
            else:
                tf_tables += (f"| {r['name']} | - | - | "
                              f"{r['total_return']:.1f}% | {r['sharpe']:.2f} | "
                              f"{r['num_trades']} |\n")
        tf_tables += "\n"

    report = f"""# 最終策略比較報告

生成時間: {now}

## 多時間框架結果

{tf_tables}
## 最佳策略
- **時間框架:** {best.get('timeframe', 'N/A')}
- **策略類型:** {best_type}
- **參數:** {best['name']}
- **預期報酬:** {best['total_return']:.1f}%
- **預期夏普:** {best['sharpe']:.2f}
- **交易次數:** {best['num_trades']}

## 建議
1. 跨時間框架測試有助於找到在不同市場條件下都表現穩定的策略
2. 報酬率最高的策略未必最穩健，需結合夏普比率和最大回撤判斷
3. ML 模型需定期重新訓練以適應市場變化
4. 建議在實盤前先用紙上交易驗證至少 2 週
5. 注意：當前市場可能處於特定趨勢，策略需在多種市場環境中驗證
"""
    return report


# ── Main ───────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("最終策略優化器 — 多時間框架分析")
    print("=" * 60)

    tf_configs = [
        ("1d", 400),   # 365+ candles
        ("4h", 1100),  # 1000+ candles
        ("1h", 2200),  # 2000+ candles
    ]

    tf_results = {}
    all_results = []

    for tf, limit in tf_configs:
        print(f"\n{'=' * 60}")
        print(f"Timeframe: {tf}")
        print(f"{'=' * 60}")

        # Fetch data
        df = fetch_ohlcv("BTC/USDT", tf, limit)

        # Build features
        print(f"  Building features...")
        features, labels = build_features(df)
        print(f"  Features: {len(features.columns)} cols, {len(features)} rows")

        # Train/test split (80/20)
        split_idx = int(len(features) * 0.8)

        # Rule strategies
        print(f"\n  Rule-Based Strategies:")
        rule_results = test_rule_strategies(features.iloc[split_idx:], tf)
        for r in rule_results:
            print(f"    {r['name']}: return={r['total_return']:.1f}%, sharpe={r['sharpe']:.2f}, trades={r['num_trades']}")

        # ML strategies
        print(f"\n  ML Model Strategies:")
        feature_columns = [c for c in features.columns
                           if c not in ("close", "momentum_score", "momentum_delta", "momentum_acceleration")]
        ml_results = test_ml_models(features, labels, feature_columns, split_idx, tf)
        for r in ml_results:
            print(f"    {r['name']}: return={r['total_return']:.1f}%, sharpe={r['sharpe']:.2f}, trades={r['num_trades']}")

        # Tag with timeframe
        for r in rule_results + ml_results:
            r["timeframe"] = tf

        tf_results[tf] = rule_results + ml_results
        all_results.extend(rule_results + ml_results)

    # Generate report
    print(f"\n{'=' * 60}")
    print("Generating report...")
    print(f"{'=' * 60}")
    report = generate_report(tf_results, all_results)
    report_path = "final_strategy_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved: {report_path}")

    # Final summary
    best = max(all_results, key=lambda x: x["total_return"])
    print(f"\n{'=' * 60}")
    print("BEST STRATEGY ACROSS ALL TIMEFRAMES")
    print(f"{'=' * 60}")
    print(f"  Timeframe: {best['timeframe']}")
    print(f"  Type: {best['type']}")
    print(f"  Name: {best['name']}")
    print(f"  Return: {best['total_return']:.1f}%")
    print(f"  Sharpe: {best['sharpe']:.2f}")
    print(f"  Trades: {best['num_trades']}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
