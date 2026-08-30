#!/usr/bin/env python3
"""Annualized return optimizer: tests rule-based and ML strategies across
1d, 4h, and 1h timeframes with 1-2 years of historical data,
calculating annualized returns and Sharpe ratios."""
import sys
sys.path.insert(0, ".")

import warnings
warnings.filterwarnings("ignore")

import ccxt
import numpy as np
import pandas as pd
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

    tf_minutes = {"1h": 60, "4h": 240, "1d": 1440}.get(timeframe, 60)
    start_offset = (limit * tf_minutes + 1440) * 60 * 1000
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

def sma(series, window):
    return series.rolling(window=window, min_periods=window).mean()

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window=window, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line

def atr(high, low, close, window=14):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=window, min_periods=window).mean()

def mfi(high, low, close, volume, window=14):
    tp = (high + low + close) / 3
    mf = tp * volume
    pos_mf = mf.where(tp > tp.shift(1), 0).rolling(window=window, min_periods=window).sum()
    neg_mf = mf.where(tp < tp.shift(1), 0).rolling(window=window, min_periods=window).sum()
    ratio = pos_mf / neg_mf.replace(0, np.nan)
    return 100 - (100 / (1 + ratio))

def obv(close, volume):
    direction = np.sign(close.diff())
    direction.iloc[0] = 0
    return (volume * direction).cumsum()

def compute_indicators(df):
    """Compute local technical indicators."""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    sma_20 = sma(close, 20)
    sma_50 = sma(close, 50)
    rsi_val = rsi(close, 14)
    macd_line, macd_signal = macd(close, 12, 26, 9)
    atr_val = atr(high, low, close, 14)
    mfi_val = mfi(high, low, close, volume, 14)
    obv_val = obv(close, volume)

    return pd.DataFrame({
        "SMA_20": sma_20,
        "SMA_50": sma_50,
        "RSI": rsi_val,
        "MACD": macd_line,
        "ATR": atr_val,
        "MFI": mfi_val,
        "OBV": obv_val,
    }, index=df.index)


def build_features(df):
    """Build feature matrix with momentum score and labels."""
    indicators = compute_indicators(df)
    indicators["close"] = df["close"]

    indicators["momentum_score"] = (
        (indicators["RSI"] - 50) / 50 * 0.3
        + indicators["MACD"] / indicators["close"].abs() * 100 * 0.1
        + (indicators["close"] / indicators["SMA_20"] - 1) * 0.4
        + (indicators["close"] / indicators["SMA_50"] - 1) * 0.2
    )
    indicators["momentum_delta"] = indicators["momentum_score"].diff()
    indicators["momentum_acceleration"] = indicators["momentum_delta"].diff()

    n_bars = 5
    future_return = df["close"].shift(-n_bars) / df["close"] - 1
    labels = (future_return > 0).astype(int)

    valid_idx = indicators.dropna().index.intersection(labels.dropna().index)
    features = indicators.loc[valid_idx]
    labels = labels.loc[valid_idx]
    return features, labels


# ── Annualized Return Calculation ──────────────────────────────

def calc_annualized_metrics(equity_curve, timeframe):
    """Calculate annualized return and Sharpe from equity curve."""
    if len(equity_curve) < 2:
        return 0.0, 0.0, 0, 0.0

    initial = equity_curve[0]
    final = equity_curve[-1]
    total_return = (final - initial) / initial if initial > 0 else 0.0

    tf_bars_per_day = {"1d": 1, "4h": 6, "1h": 24}
    bars_per_day = tf_bars_per_day.get(timeframe, 24)
    trading_days = len(equity_curve) / bars_per_day
    if trading_days <= 0:
        return total_return * 100, 0.0, int(trading_days), 0.0

    ann_return = (1 + total_return) ** (365 / trading_days) - 1

    # Daily returns for Sharpe
    eq = np.array(equity_curve)
    returns = np.diff(eq) / eq[:-1]
    returns = returns[np.isfinite(returns)]
    if len(returns) < 2:
        return total_return * 100, ann_return * 100, int(trading_days), 0.0

    bars_per_year = bars_per_day * 365
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(bars_per_year) if np.std(returns) > 0 else 0.0

    return total_return * 100, ann_return * 100, int(trading_days), sharpe


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
        total_ret, ann_ret, trading_days, ann_sharpe = calc_annualized_metrics(
            bt_result.equity_curve, timeframe
        )
        results.append({
            "name": name,
            "type": "Rule",
            "buy_th": buy_th,
            "sell_th": sell_th,
            "total_return": total_ret,
            "annualized_return": ann_ret,
            "annualized_sharpe": ann_sharpe,
            "trading_days": trading_days,
            "num_trades": bt_result.total_trades,
            "max_drawdown": bt_result.max_drawdown_pct,
            "win_rate": bt_result.win_rate,
        })
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
            total_ret, ann_ret, trading_days, ann_sharpe = calc_annualized_metrics(
                bt_result.equity_curve, timeframe
            )
            results.append({
                "name": f"{model_name} (th={th})",
                "type": "ML",
                "buy_th": "-",
                "sell_th": "-",
                "total_return": total_ret,
                "annualized_return": ann_ret,
                "annualized_sharpe": ann_sharpe,
                "trading_days": trading_days,
                "num_trades": bt_result.total_trades,
                "max_drawdown": bt_result.max_drawdown_pct,
                "win_rate": bt_result.win_rate,
                "test_accuracy": round(test_acc, 4),
            })

    return results


# ── Report Generation ──────────────────────────────────────────

def generate_report(tf_results, all_results):
    """Generate the annualized return report."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Data summary
    data_rows = ""
    for tf in ["1d", "4h", "1h"]:
        info = tf_results.get(tf, {})
        meta = info.get("meta", {})
        data_rows += (f"| {tf} | {meta.get('total', 'N/A')} | "
                      f"{meta.get('train', 'N/A')} | {meta.get('test', 'N/A')} | "
                      f"{meta.get('period', 'N/A')} |\n")

    # Per-timeframe tables
    tf_tables = ""
    for tf in ["1d", "4h", "1h"]:
        tf_label = {"1d": "1d (日線)", "4h": "4h (4小時)", "1h": "1h (小時)"}[tf]
        rows = sorted(
            tf_results.get(tf, {}).get("results", []),
            key=lambda x: x["annualized_return"],
            reverse=True,
        )
        tf_tables += f"## {tf_label} 年化報酬比較\n\n"
        tf_tables += "| 排名 | 策略 | Buy | Sell | 總報酬% | 年化報酬% | 年化Sharpe | 交易次數 |\n"
        tf_tables += "|------|------|-----|------|---------|-----------|------------|----------|\n"
        for rank, r in enumerate(rows, 1):
            tf_tables += (f"| {rank} | {r['name']} | {r['buy_th']} | {r['sell_th']} | "
                          f"{r['total_return']:.1f}% | {r['annualized_return']:.1f}% | "
                          f"{r['annualized_sharpe']:.2f} | {r['num_trades']} |\n")
        tf_tables += "\n"

    # Best strategy
    best = max(all_results, key=lambda x: x["annualized_return"])
    best_type = "ML" if best["type"] == "ML" else "規則"

    report = f"""# 年化報酬策略比較報告

生成時間: {now}

## 資料摘要
| 時間框架 | 總筆數 | 訓練集 | 測試集 | 資料期間 |
|----------|--------|--------|--------|----------|
{data_rows}
{tf_tables}
## 最佳策略
- **時間框架:** {best.get('timeframe', 'N/A')}
- **策略類型:** {best_type}
- **參數:** {best['name']}
- **總報酬:** {best['total_return']:.1f}%
- **年化報酬:** {best['annualized_return']:.1f}%
- **年化夏普:** {best['annualized_sharpe']:.2f}

## 建議
1. 年化報酬更能反映策略的真實表現能力，消除了不同測試時長的偏差
2. 年化夏普比率 > 1 表示良好的風險調整後報酬
3. 跨時間框架驗證有助於識別在不同市場條件下都穩健的策略
4. ML 模型需定期重新訓練以適應市場變化
5. 建議在實盤前先用紙上交易驗證至少 2 週
"""
    return report


# ── Main ───────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("年化報酬策略優化器 — 1-2年歷史數據")
    print("=" * 60)

    # Fetch larger datasets for annualized analysis
    tf_configs = [
        ("1d", 750),    # ~2 years
        ("4h", 4000),   # ~667 days ~1.8 years
        ("1h", 8000),   # ~333 days ~11 months
    ]

    tf_results = {}
    all_results = []

    for tf, limit in tf_configs:
        print(f"\n{'=' * 60}")
        print(f"Timeframe: {tf} (target: {limit} candles)")
        print(f"{'=' * 60}")

        df = fetch_ohlcv("BTC/USDT", tf, limit)

        print(f"  Building features...")
        features, labels = build_features(df)
        print(f"  Features: {len(features.columns)} cols, {len(features)} rows")

        split_idx = int(len(features) * 0.8)
        train_days = int(split_idx / {"1d": 1, "4h": 6, "1h": 24}.get(tf, 24))
        test_days = len(features) - split_idx
        test_days_cal = int(test_days / {"1d": 1, "4h": 6, "1h": 24}.get(tf, 24))

        period_str = f"{df.index[0].strftime('%Y-%m')} ~ {df.index[-1].strftime('%Y-%m')}"

        meta = {
            "total": len(features),
            "train": f"{train_days}天",
            "test": f"{test_days_cal}天",
            "period": period_str,
        }

        # Rule strategies
        print(f"\n  Rule-Based Strategies:")
        rule_results = test_rule_strategies(features.iloc[split_idx:], tf)
        for r in rule_results:
            print(f"    {r['name']}: total={r['total_return']:.1f}%, "
                  f"annual={r['annualized_return']:.1f}%, "
                  f"sharpe={r['annualized_sharpe']:.2f}, trades={r['num_trades']}")

        # ML strategies
        print(f"\n  ML Model Strategies:")
        feature_columns = [c for c in features.columns
                           if c not in ("close", "momentum_score", "momentum_delta", "momentum_acceleration")]
        ml_results = test_ml_models(features, labels, feature_columns, split_idx, tf)
        for r in ml_results:
            print(f"    {r['name']}: total={r['total_return']:.1f}%, "
                  f"annual={r['annualized_return']:.1f}%, "
                  f"sharpe={r['annualized_sharpe']:.2f}, trades={r['num_trades']}")

        for r in rule_results + ml_results:
            r["timeframe"] = tf

        tf_results[tf] = {"meta": meta, "results": rule_results + ml_results}
        all_results.extend(rule_results + ml_results)

    # Generate report
    print(f"\n{'=' * 60}")
    print("Generating report...")
    print(f"{'=' * 60}")
    report = generate_report(tf_results, all_results)
    report_path = "annualized_strategy_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved: {report_path}")

    # Final summary
    best = max(all_results, key=lambda x: x["annualized_return"])
    print(f"\n{'=' * 60}")
    print("BEST STRATEGY (ANNUALIZED)")
    print(f"{'=' * 60}")
    print(f"  Timeframe: {best['timeframe']}")
    print(f"  Type: {best['type']}")
    print(f"  Name: {best['name']}")
    print(f"  Total Return: {best['total_return']:.1f}%")
    print(f"  Annualized Return: {best['annualized_return']:.1f}%")
    print(f"  Annualized Sharpe: {best['annualized_sharpe']:.2f}")
    print(f"  Trades: {best['num_trades']}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
