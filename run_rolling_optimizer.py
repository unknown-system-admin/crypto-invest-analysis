#!/usr/bin/env python3
"""Rolling window optimizer: retrains models periodically and generates
more trading signals via adaptive strategies."""
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
        "MACD_signal": macd_signal,
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


# ── Metrics ────────────────────────────────────────────────────

def calc_robust_metrics(equity_curve, trades, timeframe="1h"):
    """Calculate comprehensive trading metrics."""
    if len(equity_curve) < 2:
        return _empty_metrics()

    equity = np.array(equity_curve)
    returns = np.diff(equity) / equity[:-1]
    returns = returns[np.isfinite(returns)]

    total_return = (equity[-1] - equity[0]) / equity[0] * 100

    # Annualization factor
    bars_per_year = {"1h": 24 * 365, "4h": 6 * 365, "1d": 365}.get(timeframe, 24 * 365)

    # Sharpe
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(bars_per_year)
    else:
        sharpe = 0.0

    # Sortino
    downside = returns[returns < 0]
    if len(downside) > 0 and np.std(downside) > 0:
        sortino = np.mean(returns) / np.std(downside) * np.sqrt(bars_per_year)
    else:
        sortino = 0.0

    # Max drawdown
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / peak * 100
    max_dd = np.max(dd) if len(dd) > 0 else 0.0

    # Annualized return
    trading_hours = len(equity)
    years = trading_hours / bars_per_year
    ann_return = ((1 + total_return / 100) ** (1 / years) - 1) * 100 if years > 0 else 0.0

    # Trade metrics
    sell_trades = [t for t in trades if t.get("action") == "sell" and "pnl" in t]
    total_trades = len(sell_trades)
    wins = sum(1 for t in sell_trades if t["pnl"] > 0)
    total_profit = sum(t["pnl"] for t in sell_trades if t["pnl"] > 0)
    total_loss = sum(abs(t["pnl"]) for t in sell_trades if t["pnl"] < 0)

    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
    profit_factor = (total_profit / total_loss) if total_loss > 0 else 0.0

    # Average trade duration (approximate from trade list)
    buy_times = [i for i, t in enumerate(trades) if t["action"] == "buy"]
    sell_times = [i for i, t in enumerate(trades) if t["action"] == "sell"]
    durations = []
    for b in buy_times:
        matching_sells = [s for s in sell_times if s > b]
        if matching_sells:
            durations.append(matching_sells[0] - b)
    avg_duration = np.mean(durations) if durations else 0.0

    return {
        "total_return": round(total_return, 2),
        "annualized_return": round(ann_return, 2),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "max_drawdown": round(max_dd, 2),
        "win_rate": round(win_rate, 1),
        "profit_factor": round(profit_factor, 2),
        "total_trades": total_trades,
        "avg_duration": round(avg_duration, 1),
    }


def _empty_metrics():
    return {
        "total_return": 0.0, "annualized_return": 0.0, "sharpe": 0.0,
        "sortino": 0.0, "max_drawdown": 0.0, "win_rate": 0.0,
        "profit_factor": 0.0, "total_trades": 0, "avg_duration": 0.0,
    }


# ── Position Sizing ────────────────────────────────────────────

def kelly_criterion(win_rate_pct, avg_win, avg_loss):
    """Kelly criterion for optimal position sizing."""
    if avg_loss == 0 or win_rate_pct == 0:
        return 0.0
    w = win_rate_pct / 100
    r = avg_win / avg_loss if avg_loss > 0 else 1.0
    kelly = w - (1 - w) / r
    return max(0.0, min(kelly, 0.25))  # Cap at 25%


# ── Rolling Backtest Engine ────────────────────────────────────

class RollingBacktester:
    """Simplified backtester for rolling window evaluation."""

    def __init__(self, initial_capital=10000.0, fee_rate=0.001, slippage=0.0005,
                 position_pct=1.0, risk_pct=0.02):
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.position_pct = position_pct
        self.risk_pct = risk_pct

    def run(self, features_df, signals_df, sizing_method="fixed"):
        """Run backtest with given features and signals."""
        cash = self.initial_capital
        position = 0.0
        entry_price = 0.0
        equity_curve = []
        trades = []

        for i, (idx, row) in enumerate(features_df.iterrows()):
            price = row.get("close", 0)
            if price == 0 or idx not in signals_df.index:
                equity_curve.append(cash + position * price)
                continue

            signal = signals_df.loc[idx, "signal"]
            confidence = signals_df.loc[idx, "confidence"]

            # Update equity
            equity = cash + position * price
            equity_curve.append(equity)

            if signal == 1 and position == 0:
                # Calculate position size (account for fees in cost)
                total_cost_factor = (1 + self.slippage) * (1 + self.fee_rate)
                if sizing_method == "fixed":
                    qty = (cash * self.position_pct) / (price * total_cost_factor)
                elif sizing_method == "kelly":
                    kelly_frac = kelly_criterion(
                        signals_df.loc[:idx, "signal"].value_counts().get(1, 0) /
                        max(len(signals_df.loc[:idx]), 1) * 100,
                        0.02, 0.01
                    )
                    qty = (cash * max(kelly_frac, 0.05)) / (price * total_cost_factor)
                elif sizing_method == "risk":
                    stop_distance = price * 0.02
                    risk_amount = cash * self.risk_pct
                    qty = risk_amount / stop_distance if stop_distance > 0 else 0
                else:
                    qty = (cash * self.position_pct) / (price * total_cost_factor)

                cost = qty * price * total_cost_factor
                if cost <= cash and qty > 0:
                    cash -= cost
                    position = qty
                    entry_price = price * (1 + self.slippage)
                    trades.append({"action": "buy", "price": entry_price, "quantity": qty, "time": idx})

            elif signal == -1 and position > 0:
                exit_price = price * (1 - self.slippage)
                proceeds = position * exit_price * (1 - self.fee_rate)
                pnl = proceeds - position * entry_price
                cash += proceeds
                trades.append({
                    "action": "sell", "price": exit_price, "quantity": position,
                    "pnl": pnl, "time": idx
                })
                position = 0.0

        # Close any open position at the end
        if position > 0:
            last_price = features_df.iloc[-1]["close"]
            exit_price = last_price * (1 - self.slippage)
            proceeds = position * exit_price * (1 - self.fee_rate)
            pnl = proceeds - position * entry_price
            cash += proceeds
            trades.append({
                "action": "sell", "price": exit_price, "quantity": position,
                "pnl": pnl, "time": features_df.index[-1]
            })
            position = 0.0

        return equity_curve, trades


# ── Signal Generation Methods ──────────────────────────────────

def momentum_signals(features, buy_th=0.01, sell_th=-0.01):
    """Momentum rule signals: buy/sell when momentum crosses thresholds."""
    signals = pd.DataFrame(index=features.index)
    signals["signal"] = 0
    signals["confidence"] = 0.0

    for i in range(1, len(features)):
        score = features.iloc[i]["momentum_score"]
        delta = features.iloc[i]["momentum_delta"]

        if score > buy_th and delta > 0:
            signals.iloc[i, signals.columns.get_loc("signal")] = 1
            signals.iloc[i, signals.columns.get_loc("confidence")] = min(abs(score) * 10, 1.0)
        elif score < sell_th and delta < 0:
            signals.iloc[i, signals.columns.get_loc("signal")] = -1
            signals.iloc[i, signals.columns.get_loc("confidence")] = min(abs(score) * 10, 1.0)

    return signals


def ml_rolling_signals(features, labels, feature_columns, train_window=500,
                       retrain_every=100, test_period=100):
    """ML rolling signals: retrain every N candles, predict next M."""
    signals = pd.DataFrame(index=features.index)
    signals["signal"] = 0
    signals["confidence"] = 0.0

    n = len(features)
    start = train_window

    while start + test_period <= n:
        # Train on latest train_window candles
        train_start = max(0, start - train_window)
        X_train = features.iloc[train_start:start][feature_columns].values
        y_train = labels.iloc[train_start:start].values

        # Test on next test_period candles
        test_end = min(start + test_period, n)
        X_test = features.iloc[start:test_end][feature_columns].values

        if len(X_train) < 50 or len(X_test) == 0:
            start += retrain_every
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = RandomForestClassifier(
            n_estimators=100, max_depth=7, min_samples_split=5, random_state=42
        )
        model.fit(X_train_s, y_train)

        proba = model.predict_proba(X_test_s)
        buy_probs = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]

        for j, prob in enumerate(buy_probs):
            idx = features.index[start + j]
            if prob >= 0.5:
                signals.loc[idx, "signal"] = 1
                signals.loc[idx, "confidence"] = prob
            elif prob <= 0.3:
                signals.loc[idx, "signal"] = -1
                signals.loc[idx, "confidence"] = 1 - prob

        start += retrain_every

    return signals


def ml_momentum_hybrid_signals(features, labels, feature_columns, buy_th=0.01,
                                sell_th=-0.01, train_window=500, retrain_every=100,
                                test_period=100):
    """ML + Momentum hybrid: both must agree to trade."""
    ml_sigs = ml_rolling_signals(features, labels, feature_columns,
                                  train_window, retrain_every, test_period)
    mom_sigs = momentum_signals(features, buy_th, sell_th)

    signals = pd.DataFrame(index=features.index)
    signals["signal"] = 0
    signals["confidence"] = 0.0

    for idx in features.index:
        ml_sig = ml_sigs.loc[idx, "signal"] if idx in ml_sigs.index else 0
        mom_sig = mom_sigs.loc[idx, "signal"] if idx in mom_sigs.index else 0

        # Both must agree
        if ml_sig == 1 and mom_sig == 1:
            signals.loc[idx, "signal"] = 1
            signals.loc[idx, "confidence"] = max(
                ml_sigs.loc[idx, "confidence"], mom_sigs.loc[idx, "confidence"]
            )
        elif ml_sig == -1 and mom_sig == -1:
            signals.loc[idx, "signal"] = -1
            signals.loc[idx, "confidence"] = max(
                ml_sigs.loc[idx, "confidence"], mom_sigs.loc[idx, "confidence"]
            )

    return signals


# ── Report Generation ──────────────────────────────────────────

def generate_report(period_results, strategy_results, best_strategy, n_windows):
    """Generate the rolling strategy report in Traditional Chinese."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Period performance table
    period_rows = ""
    for i, pr in enumerate(period_results, 1):
        period_rows += (
            f"| {i} | {pr['start'].strftime('%Y-%m')} | {pr['end'].strftime('%Y-%m')} "
            f"| {pr['return']:.1f}% | {pr['trades']} | {pr['win_rate']:.0f}% |\n"
        )

    # Strategy comparison table
    strat_rows = ""
    for name, m in strategy_results.items():
        strat_rows += (
            f"| {name} | {m['total_return']:.1f}% | {m['annualized_return']:.1f}% "
            f"| {m['sharpe']:.2f} | {m['sortino']:.2f} | {m['max_drawdown']:.1f}% "
            f"| {m['total_trades']} | {m['win_rate']:.0f}% | {m['profit_factor']:.2f} |\n"
        )

    report = f"""# 滾動窗口策略報告

生成時間: {now}

## 資料摘要
| 項目 | 數值 |
|------|------|
| 時間框架 | 1h |
| 總筆數 | {sum(pr['candles'] for pr in period_results)} |
| 訓練窗口 | 500 candles (~21天) |
| 重訓練頻率 | 100 candles (~4天) |
| 測試期數 | {n_windows} 期 |

## 各期表現
| 期間 | 開始 | 結束 | 報酬% | 交易次數 | 勝率 |
|------|------|------|-------|----------|------|
{period_rows}
## 策略比較
| 策略 | 總報酬% | 年化% | Sharpe | Sortino | 最大回撤% | 交易次數 | 勝率 | 盈虧比 |
|------|---------|-------|--------|---------|-----------|----------|------|--------|
{strat_rows}
## 最佳策略
- **策略類型:** {best_strategy['name']}
- **總報酬:** {best_strategy['total_return']:.1f}%
- **年化報酬:** {best_strategy['annualized_return']:.1f}%
- **Sharpe:** {best_strategy['sharpe']:.2f}
- **交易次數:** {best_strategy['total_trades']}
- **勝率:** {best_strategy['win_rate']:.0f}%

## 建議
1. 滾動窗口能適應市場變化，避免過擬合歷史資料
2. 年化報酬 > 15% 且 Sharpe > 1 表示策略穩健
3. 最大回撤 < 20% 表示風險可控
4. 建議每 4-7 天重新訓練模型以適應新市場環境
5. 實盤前建議用紙上交易驗證至少 2 週
6. 注意交易成本對高頻策略的侵蝕
"""
    return report


# ── Main ───────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("滾動窗口策略優化器 — 自適應重訓練")
    print("=" * 60)

    # Fetch data
    SYMBOL = "BTC/USDT"
    TIMEFRAME = "1h"
    LIMIT = 8000

    df = fetch_ohlcv(SYMBOL, TIMEFRAME, LIMIT)

    # Build features
    print("\nBuilding features...")
    features, labels = build_features(df)
    print(f"  Features: {len(features.columns)} cols, {len(features)} rows")

    feature_columns = [c for c in features.columns
                       if c not in ("close", "momentum_score", "momentum_delta",
                                    "momentum_acceleration")]

    # Rolling window parameters
    TRAIN_WINDOW = 500
    RETRAIN_EVERY = 100
    TEST_PERIOD = 100

    n = len(features)
    period_results = []

    print(f"\nRolling windows: train={TRAIN_WINDOW}, retrain={RETRAIN_EVERY}, test={TEST_PERIOD}")
    print(f"Total candles available: {n}")

    # Continuous equity tracking: maintain cash+position across windows
    INITIAL_CAPITAL = 10000.0
    cash_mom = INITIAL_CAPITAL
    pos_mom = 0.0
    entry_mom = 0.0
    cash_ml = INITIAL_CAPITAL
    pos_ml = 0.0
    entry_ml = 0.0
    cash_hybrid = INITIAL_CAPITAL
    pos_hybrid = 0.0
    entry_hybrid = 0.0

    all_trades_momentum = []
    all_trades_ml = []
    all_trades_hybrid = []
    all_equity_momentum = [INITIAL_CAPITAL]
    all_equity_ml = [INITIAL_CAPITAL]
    all_equity_hybrid = [INITIAL_CAPITAL]

    FEE = 0.001
    SLIP = 0.0005
    COST_FACTOR = (1 + SLIP) * (1 + FEE)

    start = TRAIN_WINDOW
    window_num = 0

    while start + TEST_PERIOD <= n:
        window_num += 1
        test_start = start
        test_end = min(start + TEST_PERIOD, n)

        test_features = features.iloc[test_start:test_end]
        test_labels = labels.iloc[test_start:test_end]
        test_start_date = test_features.index[0]
        test_end_date = test_features.index[-1]

        print(f"\n  Window {window_num}: {test_start_date.strftime('%Y-%m-%d')} ~ "
              f"{test_end_date.strftime('%Y-%m-%d')} ({test_end - test_start} candles)")

        # Strategy A: Momentum Rule
        mom_sigs = momentum_signals(test_features, buy_th=0.01, sell_th=-0.01)

        # Strategy B: ML Rolling (train on preceding data)
        train_start = max(0, start - TRAIN_WINDOW)
        train_features = features.iloc[train_start:start]
        train_labels = labels.iloc[train_start:start]

        ml_sigs = pd.DataFrame(index=test_features.index)
        ml_sigs["signal"] = 0
        ml_sigs["confidence"] = 0.0

        if len(train_features) >= 50:
            X_train = train_features[feature_columns].values
            y_train = train_labels.values
            X_test = test_features[feature_columns].values

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            model = RandomForestClassifier(
                n_estimators=100, max_depth=7, min_samples_split=5, random_state=42
            )
            model.fit(X_train_s, y_train)

            proba = model.predict_proba(X_test_s)
            buy_probs = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]

            for j, prob in enumerate(buy_probs):
                idx = test_features.index[j]
                if prob >= 0.5:
                    ml_sigs.loc[idx, "signal"] = 1
                    ml_sigs.loc[idx, "confidence"] = prob
                elif prob <= 0.4:
                    ml_sigs.loc[idx, "signal"] = -1
                    ml_sigs.loc[idx, "confidence"] = 1 - prob

        # Strategy C: ML + Momentum Hybrid
        hybrid_sigs = pd.DataFrame(index=test_features.index)
        hybrid_sigs["signal"] = 0
        hybrid_sigs["confidence"] = 0.0

        for idx in test_features.index:
            ml_sig = ml_sigs.loc[idx, "signal"] if idx in ml_sigs.index else 0
            mom_sig = mom_sigs.loc[idx, "signal"] if idx in mom_sigs.index else 0
            if ml_sig == 1 and mom_sig == 1:
                hybrid_sigs.loc[idx, "signal"] = 1
                hybrid_sigs.loc[idx, "confidence"] = max(
                    ml_sigs.loc[idx, "confidence"], mom_sigs.loc[idx, "confidence"]
                )
            elif ml_sig == -1 and mom_sig == -1:
                hybrid_sigs.loc[idx, "signal"] = -1
                hybrid_sigs.loc[idx, "confidence"] = max(
                    ml_sigs.loc[idx, "confidence"], mom_sigs.loc[idx, "confidence"]
                )

        # Execute trades continuously across windows
        window_trades_mom = []
        window_trades_ml = []
        window_trades_hybrid = []

        for i, (idx, row) in enumerate(test_features.iterrows()):
            price = row["close"]
            if price == 0:
                continue

            # --- Momentum ---
            s = mom_sigs.loc[idx, "signal"] if idx in mom_sigs.index else 0
            if s == 1 and pos_mom == 0:
                qty = cash_mom / (price * COST_FACTOR)
                if qty > 0:
                    cost = qty * price * COST_FACTOR
                    cash_mom -= cost
                    pos_mom = qty
                    entry_mom = price * (1 + SLIP)
                    window_trades_mom.append({"action": "buy", "price": entry_mom, "quantity": qty, "time": idx})
            elif s == -1 and pos_mom > 0:
                exit_p = price * (1 - SLIP)
                proceeds = pos_mom * exit_p * (1 - FEE)
                pnl = proceeds - pos_mom * entry_mom
                cash_mom += proceeds
                window_trades_mom.append({"action": "sell", "price": exit_p, "quantity": pos_mom, "pnl": pnl, "time": idx})
                pos_mom = 0.0

            # --- ML ---
            s = ml_sigs.loc[idx, "signal"] if idx in ml_sigs.index else 0
            if s == 1 and pos_ml == 0:
                qty = cash_ml / (price * COST_FACTOR)
                if qty > 0:
                    cost = qty * price * COST_FACTOR
                    cash_ml -= cost
                    pos_ml = qty
                    entry_ml = price * (1 + SLIP)
                    window_trades_ml.append({"action": "buy", "price": entry_ml, "quantity": qty, "time": idx})
            elif s == -1 and pos_ml > 0:
                exit_p = price * (1 - SLIP)
                proceeds = pos_ml * exit_p * (1 - FEE)
                pnl = proceeds - pos_ml * entry_ml
                cash_ml += proceeds
                window_trades_ml.append({"action": "sell", "price": exit_p, "quantity": pos_ml, "pnl": pnl, "time": idx})
                pos_ml = 0.0

            # --- Hybrid ---
            s = hybrid_sigs.loc[idx, "signal"] if idx in hybrid_sigs.index else 0
            if s == 1 and pos_hybrid == 0:
                qty = cash_hybrid / (price * COST_FACTOR)
                if qty > 0:
                    cost = qty * price * COST_FACTOR
                    cash_hybrid -= cost
                    pos_hybrid = qty
                    entry_hybrid = price * (1 + SLIP)
                    window_trades_hybrid.append({"action": "buy", "price": entry_hybrid, "quantity": qty, "time": idx})
            elif s == -1 and pos_hybrid > 0:
                exit_p = price * (1 - SLIP)
                proceeds = pos_hybrid * exit_p * (1 - FEE)
                pnl = proceeds - pos_hybrid * entry_hybrid
                cash_hybrid += proceeds
                window_trades_hybrid.append({"action": "sell", "price": exit_p, "quantity": pos_hybrid, "pnl": pnl, "time": idx})
                pos_hybrid = 0.0

            # Record equity
            eq_m = cash_mom + pos_mom * price
            eq_ml_val = cash_ml + pos_ml * price
            eq_h = cash_hybrid + pos_hybrid * price
            all_equity_momentum.append(eq_m)
            all_equity_ml.append(eq_ml_val)
            all_equity_hybrid.append(eq_h)

        all_trades_momentum.extend(window_trades_mom)
        all_trades_ml.extend(window_trades_ml)
        all_trades_hybrid.extend(window_trades_hybrid)

        # Per-period return (from momentum trades)
        mom_buys = sum(1 for t in window_trades_mom if t["action"] == "buy")
        mom_sells = [t for t in window_trades_mom if t["action"] == "sell" and "pnl" in t]
        mom_wr = (sum(1 for t in mom_sells if t["pnl"] > 0) / len(mom_sells) * 100) if mom_sells else 0
        # Approximate period return from equity change
        eq_start = all_equity_momentum[-len(test_features) - 1]
        eq_end = all_equity_momentum[-1]
        mom_ret = (eq_end - eq_start) / eq_start * 100 if eq_start > 0 else 0

        period_results.append({
            "start": test_start_date,
            "end": test_end_date,
            "return": mom_ret,
            "trades": mom_buys,
            "win_rate": mom_wr,
            "candles": test_end - test_start,
        })

        print(f"    Momentum: {mom_ret:.1f}% ({mom_buys} buys, {len(mom_sells)} sells)")
        print(f"    ML:       buys={sum(1 for t in window_trades_ml if t['action']=='buy')}")
        print(f"    Hybrid:   buys={sum(1 for t in window_trades_hybrid if t['action']=='buy')}")

        start += RETRAIN_EVERY

    # Close any remaining positions at the end
    last_price = features.iloc[-1]["close"]
    if pos_mom > 0:
        exit_p = last_price * (1 - SLIP)
        proceeds = pos_mom * exit_p * (1 - FEE)
        pnl = proceeds - pos_mom * entry_mom
        cash_mom += proceeds
        all_trades_momentum.append({"action": "sell", "price": exit_p, "quantity": pos_mom, "pnl": pnl, "time": features.index[-1]})
        pos_mom = 0.0
    if pos_ml > 0:
        exit_p = last_price * (1 - SLIP)
        proceeds = pos_ml * exit_p * (1 - FEE)
        pnl = proceeds - pos_ml * entry_ml
        cash_ml += proceeds
        all_trades_ml.append({"action": "sell", "price": exit_p, "quantity": pos_ml, "pnl": pnl, "time": features.index[-1]})
        pos_ml = 0.0
    if pos_hybrid > 0:
        exit_p = last_price * (1 - SLIP)
        proceeds = pos_hybrid * exit_p * (1 - FEE)
        pnl = proceeds - pos_hybrid * entry_hybrid
        cash_hybrid += proceeds
        all_trades_hybrid.append({"action": "sell", "price": exit_p, "quantity": pos_hybrid, "pnl": pnl, "time": features.index[-1]})
        pos_hybrid = 0.0

    # Overall metrics
    print(f"\n{'=' * 60}")
    print("Calculating overall metrics...")

    metrics_mom = calc_robust_metrics(all_equity_momentum, all_trades_momentum, TIMEFRAME)
    metrics_ml = calc_robust_metrics(all_equity_ml, all_trades_ml, TIMEFRAME)
    metrics_hybrid = calc_robust_metrics(all_equity_hybrid, all_trades_hybrid, TIMEFRAME)

    strategy_results = {
        "動量規則": metrics_mom,
        "ML Rolling": metrics_ml,
        "ML+動量": metrics_hybrid,
    }

    # Find best strategy
    best_name = max(strategy_results, key=lambda k: strategy_results[k]["sharpe"])
    best_strategy = strategy_results[best_name].copy()
    best_strategy["name"] = best_name

    # Generate report
    print("\nGenerating report...")
    report = generate_report(period_results, strategy_results, best_strategy, window_num)
    report_path = "rolling_strategy_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report saved: {report_path}")

    # Final summary
    print(f"\n{'=' * 60}")
    print("BEST STRATEGY (ROLLING WINDOW)")
    print(f"{'=' * 60}")
    print(f"  Strategy:     {best_strategy['name']}")
    print(f"  Total Return: {best_strategy['total_return']:.1f}%")
    print(f"  Annualized:   {best_strategy['annualized_return']:.1f}%")
    print(f"  Sharpe:       {best_strategy['sharpe']:.2f}")
    print(f"  Sortino:      {best_strategy['sortino']:.2f}")
    print(f"  Max Drawdown: {best_strategy['max_drawdown']:.1f}%")
    print(f"  Trades:       {best_strategy['total_trades']}")
    print(f"  Win Rate:     {best_strategy['win_rate']:.0f}%")
    print(f"  Profit Factor:{best_strategy['profit_factor']:.2f}")
    print(f"  Windows:      {window_num}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
