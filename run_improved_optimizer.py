#!/usr/bin/env python3
"""Improved strategy optimizer: more data, shorter lookback, aggressive strategies."""
import sys
sys.path.insert(0, ".")

import warnings
warnings.filterwarnings("ignore")

import ccxt
import numpy as np
import pandas as pd
import ta
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from backtest_engine.engine import BacktestEngine
from backtest_engine.rule_strategy import MomentumRuleStrategy
from backtest_engine.strategy import Strategy, Signal


# ── Data Fetching (1h, 2000+ candles) ─────────────────────────

def fetch_1h_data(symbol="BTC/USDT", limit=2500):
    """Fetch 1h OHLCV from OKX with pagination. OKX returns max ~300 per call."""
    print(f"Fetching {limit} candles of {symbol} 1h data from OKX...")
    exchange = ccxt.okx({"enableRateLimit": True})
    all_data = []
    # Start far enough back
    since = int((datetime.now() - timedelta(days=limit // 24 + 30)).timestamp() * 1000)

    while len(all_data) < limit:
        try:
            batch = exchange.fetch_ohlcv(symbol, "1h", since=since, limit=300)
        except Exception as e:
            print(f"  Fetch error: {e}, retrying...")
            import time
            time.sleep(2)
            continue
        if not batch:
            break
        all_data.extend(batch)
        since = batch[-1][0] + 1
        if len(batch) < 300:
            break
        import time
        time.sleep(0.2)

    all_data = all_data[:limit]
    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df[~df.index.duplicated(keep="first")]
    print(f"  Period: {df.index[0]} ~ {df.index[-1]} ({len(df)} rows)")
    return df


# ── Local Feature Engine (shorter SMA periods) ────────────────

def compute_local_indicators(df):
    """Compute indicators with shorter SMA (20, 50 only). Avoids 200-bar warmup."""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    sma_20 = ta.trend.sma_indicator(close, window=20)
    sma_50 = ta.trend.sma_indicator(close, window=50)
    rsi = ta.momentum.rsi(close, window=14)
    macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    macd_line = macd.macd()
    atr = ta.volatility.average_true_range(high, low, close, window=14)
    mfi = ta.volume.money_flow_index(high, low, close, volume, window=14)
    obv = ta.volume.on_balance_volume(close, volume)

    result = pd.DataFrame({
        "SMA_20": sma_20,
        "SMA_50": sma_50,
        "RSI": rsi,
        "MACD": macd_line,
        "ATR": atr,
        "MFI": mfi,
        "OBV": obv,
    }, index=df.index)
    return result


def build_optimized_features(df):
    """Build features with local engine + momentum score + binary labels."""
    indicators = compute_local_indicators(df)
    indicators["close"] = df["close"]

    # Momentum score: composite of RSI + MFI + MACD normalized
    indicators["momentum_score"] = (
        (indicators["RSI"] - 50) / 50 * 0.4
        + (indicators["MFI"] - 50) / 50 * 0.3
        + indicators["MACD"] / indicators["close"].abs() * 100 * 0.3
    )
    indicators["momentum_delta"] = indicators["momentum_score"].diff()
    indicators["momentum_acceleration"] = indicators["momentum_delta"].diff()

    # Binary label: price goes up in next 5 bars
    n_bars = 5
    future_return = df["close"].shift(-n_bars) / df["close"] - 1
    labels = (future_return > 0).astype(int)

    valid_idx = indicators.dropna().index.intersection(labels.dropna().index)
    features = indicators.loc[valid_idx]
    labels = labels.loc[valid_idx]
    return features, labels


# ── Backtest Runner ────────────────────────────────────────────

def run_backtest(strategy, features_df, timeframe="1h"):
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
    return {
        "total_return": result.total_return_pct,
        "sharpe": result.sharpe_ratio,
        "max_drawdown": result.max_drawdown_pct,
        "num_trades": result.total_trades,
        "win_rate": result.win_rate,
    }


# ── Rule-Based Strategies (aggressive thresholds) ─────────────

def test_rule_strategies(features):
    """Test ultra-aggressive to moderate rule strategies."""
    configs = [
        ("Ultra Aggressive", 0.01, -0.01),
        ("Very Aggressive", 0.02, -0.02),
        ("Aggressive", 0.03, -0.03),
        ("Moderate", 0.05, -0.05),
    ]
    results = []
    for name, buy_th, sell_th in configs:
        strategy = MomentumRuleStrategy(buy_threshold=buy_th, sell_threshold=sell_th)
        bt_result = run_backtest(strategy, features)
        m = extract_metrics(bt_result)
        m["name"] = name
        m["buy_th"] = buy_th
        m["sell_th"] = sell_th
        results.append(m)
        print(f"  {name} (buy={buy_th}, sell={sell_th}): "
              f"return={m['total_return']:.1f}%, sharpe={m['sharpe']:.2f}, "
              f"trades={m['num_trades']}, win={m['win_rate']:.1f}%")
    return results


# ── ML Model Strategies ───────────────────────────────────────

class MLThresholdStrategy(Strategy):
    """ML model with configurable buy probability threshold, auto-scales features."""
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


def test_ml_models(features, labels, feature_columns, split_idx):
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

        # Debug: check prediction distribution
        probas = model.predict_proba(X_test_scaled)
        avg_buy_prob = probas[:, 1].mean()
        print(f"  {model_name}: test_acc={test_acc:.4f}, avg_buy_prob={avg_buy_prob:.4f}")

        for th in thresholds:
            strategy = MLThresholdStrategy(model, feature_columns, scaler, buy_threshold=th)
            bt_result = run_backtest(strategy, test_features)
            m = extract_metrics(bt_result)
            m["name"] = f"{model_name} ({th})"
            m["test_accuracy"] = round(test_acc, 4)
            results.append(m)
            print(f"  {model_name} (th={th}): "
                  f"return={m['total_return']:.1f}%, sharpe={m['sharpe']:.2f}, "
                  f"trades={m['num_trades']}, acc={test_acc:.4f}")

    return results


# ── Report Generation ──────────────────────────────────────────

def generate_report(df, train_days, test_days, rule_results, ml_results):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    date_range = f"{df.index[0]} ~ {df.index[-1]}"
    total_rows = len(df)

    all_results = rule_results + ml_results
    best = max(all_results, key=lambda x: x["total_return"])

    # Rule table (sorted by return)
    rule_sorted = sorted(rule_results, key=lambda x: x["total_return"], reverse=True)
    rule_rows = ""
    for i, r in enumerate(rule_sorted, 1):
        rule_rows += (f"| {i} | {r['name']} | {r['buy_th']} | {r['sell_th']} | "
                      f"{r['total_return']:.1f}% | {r['sharpe']:.2f} | "
                      f"{r['win_rate']:.0f}% | {r['num_trades']} |\n")

    # ML table (sorted by return)
    ml_sorted = sorted(ml_results, key=lambda x: x["total_return"], reverse=True)
    ml_rows = ""
    for r in ml_sorted:
        ml_rows += (f"| {r['name']} | {r['total_return']:.1f}% | "
                    f"{r['sharpe']:.2f} | {r['win_rate']:.0f}% | "
                    f"{r['num_trades']} |\n")

    best_type = "ML" if best["name"][0:2] in ("RF", "XB") else "Rule"

    report = f"""# 改進策略比較報告

生成時間: {now}

## 資料摘要
| 項目 | 數值 |
|------|------|
| 時間框架 | 1h |
| 資料筆數 | {total_rows} |
| 訓練集 | {train_days} 天 |
| 測試集 | {test_days} 天 |

## 規則策略比較 (按報酬率排序)
| 排名 | 策略 | Buy | Sell | 報酬% | Sharpe | 勝率 | 交易次數 |
|------|------|-----|------|-------|--------|------|----------|
{rule_rows}
## ML 模型比較
| 模型 | 報酬% | Sharpe | 勝率 | 交易次數 |
|------|-------|--------|------|----------|
{ml_rows}
## 最佳策略
- **最佳策略類型:** {best_type}
- **最佳參數:** {best['name']}
- **預期報酬:** {best['total_return']:.1f}%
- **預期夏普:** {best['sharpe']:.2f}
- **交易次數:** {best['num_trades']}

## 建議
1. 報酬率最高的策略未必最穩健，需結合夏普比率判斷
2. 高頻交易策略手續費影響較大，注意實際滑點
3. ML 模型需定期重新訓練以適應市場變化
4. 建議在實盤前先用紙上交易驗證
"""

    return report


# ── Main ───────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("改進策略優化器 — 最大化報酬率")
    print("=" * 60)

    # 1. Fetch 1h data (2500 candles)
    df = fetch_1h_data("BTC/USDT", limit=2500)

    # 2. Build features with local engine
    print("\nBuilding feature matrix (local engine, shorter SMA)...")
    features, labels = build_optimized_features(df)
    print(f"  Features: {len(features.columns)} columns, {len(features)} rows")

    # 3. Train/test split (80/20)
    split_idx = int(len(features) * 0.8)
    train_days = len(features.iloc[:split_idx]) // 24
    test_days = len(features.iloc[split_idx:]) // 24
    print(f"  Train: {train_days} days ({split_idx} bars), Test: {test_days} days ({len(features) - split_idx} bars)")

    feature_columns = [c for c in features.columns if c not in ("close", "momentum_score", "momentum_delta", "momentum_acceleration")]

    # 4. Test rule strategies on test set
    print("\n" + "=" * 60)
    print("A. Rule-Based Strategies")
    print("=" * 60)
    test_features = features.iloc[split_idx:]
    rule_results = test_rule_strategies(test_features)

    # 5. Test ML models
    print("\n" + "=" * 60)
    print("B. ML Model Strategies")
    print("=" * 60)
    ml_results = test_ml_models(features, labels, feature_columns, split_idx)

    # 6. Generate report
    print("\n" + "=" * 60)
    print("Generating report...")
    print("=" * 60)
    report = generate_report(df, train_days, test_days, rule_results, ml_results)
    report_path = "improved_strategy_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved: {report_path}")

    # 7. Summary
    all_results = rule_results + ml_results
    best = max(all_results, key=lambda x: x["total_return"])
    print("\n" + "=" * 60)
    print("Best Strategy")
    print("=" * 60)
    print(f"  Name: {best['name']}")
    print(f"  Return: {best['total_return']:.1f}%")
    print(f"  Sharpe: {best['sharpe']:.2f}")
    print(f"  Trades: {best['num_trades']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
