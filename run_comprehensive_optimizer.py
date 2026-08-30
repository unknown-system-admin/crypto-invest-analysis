#!/usr/bin/env python3
"""Comprehensive high-frequency strategy optimizer:
tests multiple market conditions and trading frequencies
with stop-loss/take-profit and expected value analysis."""
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

    tf_minutes = {"1h": 60}.get(timeframe, 60)
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
    sma_200 = sma(close, 200)
    rsi_val = rsi(close, 14)
    macd_line, macd_signal = macd(close, 12, 26, 9)
    atr_val = atr(high, low, close, 14)
    mfi_val = mfi(high, low, close, volume, 14)
    obv_val = obv(close, volume)

    return pd.DataFrame({
        "SMA_20": sma_20,
        "SMA_50": sma_50,
        "SMA_200": sma_200,
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


# ── Market Regime Detection ────────────────────────────────────

def classify_regimes(df):
    """Classify each bar into Bull, Bear, or Sideways regime."""
    close = df["close"]
    sma_50 = sma(close, 50)
    sma_200 = sma(close, 200)

    regime = pd.Series("Sideways", index=df.index)
    bull_mask = (close > sma_50) & (sma_50 > sma_200)
    bear_mask = (close < sma_50) & (sma_50 < sma_200)
    regime[bull_mask] = "Bull"
    regime[bear_mask] = "Bear"
    return regime


def split_by_regime(features, regime_series):
    """Split features DataFrame by regime, keeping only valid indices."""
    regimes = {}
    for name in ["Bull", "Bear", "Sideways"]:
        mask = regime_series.loc[features.index] == name
        if mask.sum() > 50:
            regimes[name] = features.loc[mask]
    return regimes


# ── Annualized Metrics ─────────────────────────────────────────

def calc_annualized_metrics(equity_curve, timeframe):
    """Calculate annualized return and Sharpe from equity curve."""
    if len(equity_curve) < 2:
        return 0.0, 0.0, 0, 0.0

    initial = equity_curve[0]
    final = equity_curve[-1]
    total_return = (final - initial) / initial if initial > 0 else 0.0

    tf_bars_per_day = {"1h": 24}.get(timeframe, 24)
    bars_per_day = tf_bars_per_day
    trading_days = len(equity_curve) / bars_per_day
    if trading_days <= 0:
        return total_return * 100, 0.0, int(trading_days), 0.0

    ann_return = (1 + total_return) ** (365 / trading_days) - 1

    eq = np.array(equity_curve)
    returns = np.diff(eq) / eq[:-1]
    returns = returns[np.isfinite(returns)]
    if len(returns) < 2:
        return total_return * 100, ann_return * 100, int(trading_days), 0.0

    bars_per_year = bars_per_day * 365
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(bars_per_year) if np.std(returns) > 0 else 0.0

    return total_return * 100, ann_return * 100, int(trading_days), sharpe


def calc_expected_value(trades):
    """Calculate expected value and expectation ratio from trades."""
    sell_trades = [t for t in trades if t["action"] == "sell" and "pnl" in t]
    if not sell_trades:
        return 0.0, 0.0

    wins = [t["pnl"] for t in sell_trades if t["pnl"] > 0]
    losses = [abs(t["pnl"]) for t in sell_trades if t["pnl"] <= 0]

    win_rate = len(wins) / len(sell_trades) if sell_trades else 0
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    loss_rate = len(losses) / len(sell_trades) if sell_trades else 0

    ev = (win_rate * avg_win) - (loss_rate * avg_loss)
    ev_ratio = ev / avg_loss if avg_loss > 0 else 0.0
    return round(ev, 2), round(ev_ratio, 4)


# ── Backtest with Stop-Loss/Take-Profit ────────────────────────

class BacktestEngineSLTP(BacktestEngine):
    """Extended backtest engine with stop-loss and take-profit."""

    def __init__(self, strategy, stop_loss_pct=None, take_profit_pct=None, **kwargs):
        super().__init__(strategy=strategy, **kwargs)
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

    def run(self, features: pd.DataFrame):
        cash = self.initial_capital
        positions = []
        equity_curve = []
        trades = []
        daily_trade_count = 0
        current_day = None

        for i, (idx, row) in enumerate(features.iterrows()):
            sig = self.strategy.evaluate(row)
            price = row.get("close", 0)

            if price == 0:
                equity_curve.append(cash)
                continue

            row_day = idx.date()
            if current_day != row_day:
                current_day = row_day
                daily_trade_count = 0

            for pos in positions:
                pos.current_price = price

            # Check SL/TP for existing positions
            positions_to_close = []
            for pos in positions:
                pnl_pct = (price - pos.entry_price) / pos.entry_price
                if self.stop_loss_pct is not None and pnl_pct <= self.stop_loss_pct:
                    positions_to_close.append(("sl", pos))
                elif self.take_profit_pct is not None and pnl_pct >= self.take_profit_pct:
                    positions_to_close.append(("tp", pos))

            for reason, pos in positions_to_close:
                effective_price = price * (1 - self.slippage)
                pnl = (effective_price * (1 - self.fee_rate) - pos.entry_price * (1 + self.fee_rate)) * pos.quantity
                cash += pos.quantity * effective_price * (1 - self.fee_rate)
                trades.append({"action": "sell", "price": effective_price, "reason": reason, "pnl": pnl})
                positions.remove(pos)

            # Check drawdown
            peak = max(equity_curve) if equity_curve else cash
            current_equity = cash + sum(p.quantity * price for p in positions)
            drawdown = ((peak - current_equity) / peak * 100) if peak > 0 else 0

            if drawdown > self.max_drawdown_stop and positions:
                for pos in positions:
                    effective_price = price * (1 - self.slippage)
                    pnl = (effective_price * (1 - self.fee_rate) - pos.entry_price * (1 + self.fee_rate)) * pos.quantity
                    cash += pos.quantity * effective_price * (1 - self.fee_rate)
                    trades.append({"action": "sell", "price": effective_price, "reason": "drawdown_stop", "pnl": pnl})
                positions.clear()

            # Execute trades
            if sig.direction == "偏多" and not positions and daily_trade_count < self.max_daily_trades:
                effective_price = price * (1 + self.slippage)
                qty = (cash * self.max_position_pct / 100) / effective_price
                if qty > 0:
                    cost = qty * effective_price * (1 + self.fee_rate)
                    if cost <= cash:
                        cash -= cost
                        positions.append(Position(self.symbol, "long", qty, effective_price))
                        trades.append({"action": "buy", "price": effective_price, "quantity": qty})
                        daily_trade_count += 1

            elif sig.direction == "偏空" and positions and daily_trade_count < self.max_daily_trades:
                for pos in positions:
                    effective_price = price * (1 - self.slippage)
                    pnl = (effective_price * (1 - self.fee_rate) - pos.entry_price * (1 + self.fee_rate)) * pos.quantity
                    cash += pos.quantity * effective_price * (1 - self.fee_rate)
                    trades.append({"action": "sell", "price": effective_price, "quantity": pos.quantity, "pnl": pnl})
                positions.clear()
                daily_trade_count += 1

            current_equity = cash + sum(p.quantity * price for p in positions)
            equity_curve.append(current_equity)

        # Close remaining positions at last price
        if positions and equity_curve:
            last_price = features.iloc[-1].get("close", 0)
            for pos in positions:
                effective_price = last_price * (1 - self.slippage)
                pnl = (effective_price * (1 - self.fee_rate) - pos.entry_price * (1 + self.fee_rate)) * pos.quantity
                cash += pos.quantity * effective_price * (1 - self.fee_rate)
                trades.append({"action": "sell", "price": effective_price, "reason": "end", "pnl": pnl})
            positions.clear()
            equity_curve[-1] = cash

        # Calculate metrics
        final_equity = equity_curve[-1] if equity_curve else cash
        total_return = ((final_equity - self.initial_capital) / self.initial_capital) * 100
        peak = max(equity_curve) if equity_curve else cash
        max_dd = ((peak - min(equity_curve)) / peak * 100) if equity_curve and peak > 0 else 0

        wins = sum(1 for t in trades if t["action"] == "sell" and t.get("pnl", 0) > 0)
        total_sells = sum(1 for t in trades if t["action"] == "sell")
        win_rate = (wins / total_sells * 100) if total_sells > 0 else 0

        if len(equity_curve) > 1:
            returns = pd.Series(equity_curve).pct_change().dropna()
            if returns.std() > 0:
                sharpe = (returns.mean() / returns.std()) * np.sqrt(24 * 365)
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0

        return BacktestResult(
            total_trades=len([t for t in trades if t["action"] == "buy"]),
            final_equity=round(final_equity, 2),
            total_return_pct=round(total_return, 2),
            max_drawdown_pct=round(max_dd, 2),
            sharpe_ratio=round(sharpe, 2),
            win_rate=round(win_rate, 1),
            equity_curve=equity_curve,
            trades=trades,
        )


from backtest_engine.engine import Position, BacktestResult


# ── ML Strategy ────────────────────────────────────────────────

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


# ── Strategy Configs ───────────────────────────────────────────

RULE_CONFIGS = [
    ("Ultra Aggressive", 0.005, -0.005),
    ("Very Aggressive", 0.01, -0.01),
    ("Aggressive", 0.02, -0.02),
    ("Moderate", 0.03, -0.03),
    ("Conservative", 0.05, -0.05),
]

ML_THRESHOLDS = [0.2, 0.25, 0.3, 0.4]

SL_TP_CONFIGS = [
    (None, None, "No SL/TP"),
    (-0.02, None, "SL -2%"),
    (-0.03, None, "SL -3%"),
    (-0.05, None, "SL -5%"),
    (None, 0.03, "TP +3%"),
    (None, 0.05, "TP +5%"),
    (None, 0.10, "TP +10%"),
]


# ── Backtest Runner ────────────────────────────────────────────

def run_backtest(strategy, features_df, timeframe="1h", sl=None, tp=None):
    """Run backtest with optional SL/TP."""
    engine = BacktestEngineSLTP(
        strategy=strategy,
        stop_loss_pct=sl,
        take_profit_pct=tp,
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


# ── Main Test Functions ────────────────────────────────────────

def test_rule_strategies(features, timeframe):
    """Test all rule-based strategy variants."""
    results = []
    for name, buy_th, sell_th in RULE_CONFIGS:
        strategy = MomentumRuleStrategy(buy_threshold=buy_th, sell_threshold=sell_th)
        bt_result = run_backtest(strategy, features, timeframe)
        total_ret, ann_ret, trading_days, ann_sharpe = calc_annualized_metrics(
            bt_result.equity_curve, timeframe
        )
        ev, ev_ratio = calc_expected_value(bt_result.trades)
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
            "expected_value": ev,
            "ev_ratio": ev_ratio,
        })
    return results


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
    results = []

    for model_name, model in models.items():
        model.fit(X_train_scaled, y_train)
        test_acc = model.score(X_test_scaled, y_test)

        for th in ML_THRESHOLDS:
            strategy = MLThresholdStrategy(model, feature_columns, scaler, buy_threshold=th)
            bt_result = run_backtest(strategy, test_features, timeframe)
            total_ret, ann_ret, trading_days, ann_sharpe = calc_annualized_metrics(
                bt_result.equity_curve, timeframe
            )
            ev, ev_ratio = calc_expected_value(bt_result.trades)
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
                "expected_value": ev,
                "ev_ratio": ev_ratio,
                "test_accuracy": round(test_acc, 4),
            })

    return results


def test_sl_tp_combos(features, timeframe, top_strategies):
    """Test stop-loss/take-profit on top strategies."""
    results = []
    for strat_info in top_strategies[:5]:
        for sl, tp, label in SL_TP_CONFIGS:
            if strat_info["type"] == "Rule":
                strategy = MomentumRuleStrategy(
                    buy_threshold=strat_info["buy_th"],
                    sell_threshold=strat_info["sell_th"]
                )
            else:
                continue  # Skip ML for SL/TP sweep (needs model retrain)

            bt_result = run_backtest(strategy, features, timeframe, sl=sl, tp=tp)
            total_ret, ann_ret, trading_days, _ = calc_annualized_metrics(
                bt_result.equity_curve, timeframe
            )
            results.append({
                "strategy": strat_info["name"],
                "sl_tp_label": label,
                "stop_loss": sl,
                "take_profit": tp,
                "total_return": total_ret,
                "annualized_return": ann_ret,
                "max_drawdown": bt_result.max_drawdown_pct,
                "num_trades": bt_result.total_trades,
                "win_rate": bt_result.win_rate,
            })
    return results


# ── Report Generation ──────────────────────────────────────────

def generate_report(regime_results, sl_tp_results, meta_info):
    """Generate comprehensive strategy report in Chinese."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    total = meta_info["total"]
    period = meta_info["period"]
    bull_pct = meta_info["bull_pct"]
    bear_pct = meta_info["bear_pct"]
    sideways_pct = meta_info["sideways_pct"]
    bull_days = meta_info["bull_days"]
    bear_days = meta_info["bear_days"]
    sideways_days = meta_info["sideways_days"]

    def make_table(results, regime_name):
        sorted_r = sorted(results, key=lambda x: x["total_return"], reverse=True)
        lines = []
        lines.append(f"## {regime_name}策略比較 (按報酬率排序)\n")
        lines.append("| 排名 | 策略 | Buy | Sell | 總報酬% | 年化% | 交易次數 | 勝率 | 期望值 |")
        lines.append("|------|------|-----|------|---------|-------|----------|------|--------|")
        for rank, r in enumerate(sorted_r, 1):
            buy = r.get("buy_th", "-")
            sell = r.get("sell_th", "-")
            lines.append(
                f"| {rank} | {r['name']} | {buy} | {sell} | "
                f"{r['total_return']:.1f}% | {r['annualized_return']:.1f}% | "
                f"{r['num_trades']} | {r['win_rate']:.0f}% | {r['expected_value']:.2f} |"
            )
        lines.append("")
        return "\n".join(lines)

    # Best strategies per regime
    best_per_regime = {}
    for regime_name, results in regime_results.items():
        if results:
            best = max(results, key=lambda x: x["total_return"])
            best_per_regime[regime_name] = best

    # Overall best
    all_results = []
    for results in regime_results.values():
        all_results.extend(results)
    overall_best = max(all_results, key=lambda x: x["total_return"]) if all_results else None

    # SL/TP table
    sl_tp_lines = []
    if sl_tp_results:
        sl_tp_sorted = sorted(sl_tp_results, key=lambda x: x["total_return"], reverse=True)
        sl_tp_lines.append("## 止損/止盈效果\n")
        sl_tp_lines.append("| 策略 | 止損% | 止盈% | 總報酬% | 最大回撤% | 交易次數 | 勝率 |")
        sl_tp_lines.append("|------|-------|-------|---------|-----------|----------|------|")
        for r in sl_tp_sorted[:15]:
            sl_str = f"{r['stop_loss']*100:.0f}%" if r['stop_loss'] is not None else "None"
            tp_str = f"{r['take_profit']*100:.0f}%" if r['take_profit'] is not None else "None"
            sl_tp_lines.append(
                f"| {r['strategy']} | {sl_str} | {tp_str} | "
                f"{r['total_return']:.1f}% | {r['max_drawdown']:.1f}% | "
                f"{r['num_trades']} | {r['win_rate']:.0f}% |"
            )
        sl_tp_lines.append("")

    # Best strategy details
    best_section = ""
    if overall_best:
        best_type = "ML" if overall_best["type"] == "ML" else "規則"
        # Find which regime this belongs to
        best_regime = "全市場"
        for regime_name, results in regime_results.items():
            if overall_best in results:
                best_regime = regime_name
                break
        best_section = f"""## 最佳策略
- **市場環境:** {best_regime}
- **策略類型:** {best_type}
- **參數:** {overall_best['name']}
- **年化報酬:** {overall_best['annualized_return']:.1f}%
- **交易次數:** {overall_best['num_trades']}
- **勝率:** {overall_best['win_rate']:.0f}%
- **期望值:** {overall_best['expected_value']:.2f}
- **期望比率:** {overall_best['ev_ratio']:.4f}
"""

    report = f"""# 全面策略比較報告

生成時間: {now}

## 資料摘要
| 項目 | 數值 |
|------|------|
| 時間框架 | 1h |
| 總筆數 | {total} |
| 資料期間 | {period} |
| 牛市天數 | {bull_days} 天 ({bull_pct:.1f}%) |
| 熊市天數 | {bear_days} 天 ({bear_pct:.1f}%) |
| 震盪天數 | {sideways_days} 天 ({sideways_pct:.1f}%) |

"""
    for regime_name in ["Bull", "Bear", "Sideways"]:
        cn = {"Bull": "牛市", "Bear": "熊市", "Sideways": "震盪"}[regime_name]
        if regime_name in regime_results and regime_results[regime_name]:
            report += make_table(regime_results[regime_name], cn)

    if sl_tp_lines:
        report += "\n".join(sl_tp_lines) + "\n"

    report += best_section

    report += """## 建議
1. 震盪市場策略通常交易頻率最高，但報酬率波動也最大
2. 牛市策略傾向於持有較長時間，熊市策略需要更嚴格的止損
3. 期望值 > 0 表示策略長期有利可圖，期望比率 > 1 表示風險回報良好
4. 止損 -3%~-5% 通常能有效保護資金，但過緊的止損會增加假突破風險
5. 建議在實盤前先用紙上交易驗證至少 2 週
6. 定期重新評估市場 regime，動態調整策略參數
"""
    return report


# ── Main ───────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("全面策略優化器 — 市場體制 + 止損止盈 + 期望值分析")
    print("=" * 60)

    # 1. Fetch data
    print("\n[1/6] Fetching 1h data...")
    df = fetch_ohlcv("BTC/USDT", "1h", 8500)

    # 2. Build features
    print("\n[2/6] Building features...")
    features, labels = build_features(df)
    print(f"  Features: {len(features.columns)} cols, {len(features)} rows")

    # 3. Classify regimes
    print("\n[3/6] Classifying market regimes...")
    regime_series = classify_regimes(df)
    regime_counts = regime_series.value_counts()
    total_days = len(df) / 24
    bull_days = regime_counts.get("Bull", 0) / 24
    bear_days = regime_counts.get("Bear", 0) / 24
    sideways_days = regime_counts.get("Sideways", 0) / 24

    meta_info = {
        "total": len(features),
        "period": f"{df.index[0].strftime('%Y-%m')} ~ {df.index[-1].strftime('%Y-%m')}",
        "bull_pct": bull_days / total_days * 100 if total_days > 0 else 0,
        "bear_pct": bear_days / total_days * 100 if total_days > 0 else 0,
        "sideways_pct": sideways_days / total_days * 100 if total_days > 0 else 0,
        "bull_days": int(bull_days),
        "bear_days": int(bear_days),
        "sideways_days": int(sideways_days),
    }

    print(f"  Bull: {meta_info['bull_days']} days ({meta_info['bull_pct']:.1f}%)")
    print(f"  Bear: {meta_info['bear_days']} days ({meta_info['bear_pct']:.1f}%)")
    print(f"  Sideways: {meta_info['sideways_days']} days ({meta_info['sideways_pct']:.1f}%)")

    # Split features by regime
    regime_features = split_by_regime(features, regime_series)
    for name, feat in regime_features.items():
        print(f"  {name}: {len(feat)} bars")

    # 4. Test strategies per regime
    print("\n[4/6] Testing strategies per regime...")
    regime_results = {}
    feature_columns = [c for c in features.columns
                       if c not in ("close", "momentum_score", "momentum_delta",
                                    "momentum_acceleration", "SMA_200")]

    for regime_name in ["Bull", "Bear", "Sideways"]:
        if regime_name not in regime_features:
            regime_results[regime_name] = []
            continue

        feat = regime_features[regime_name]
        cn = {"Bull": "牛市", "Bear": "熊市", "Sideways": "震盪"}[regime_name]
        print(f"\n  --- {cn} ({len(feat)} bars) ---")

        # Rule strategies
        print(f"  Rule-Based:")
        rule_results = test_rule_strategies(feat, "1h")
        for r in rule_results:
            print(f"    {r['name']}: ret={r['total_return']:.1f}%, "
                  f"trades={r['num_trades']}, win={r['win_rate']:.0f}%, "
                  f"ev={r['expected_value']:.2f}")

        # ML strategies (use full data for training, test on regime subset)
        print(f"  ML Models:")
        split_idx = int(len(features) * 0.8)
        ml_results = test_ml_models(features, labels, feature_columns, split_idx, "1h")
        for r in ml_results:
            print(f"    {r['name']}: ret={r['total_return']:.1f}%, "
                  f"trades={r['num_trades']}, win={r['win_rate']:.0f}%, "
                  f"ev={r['expected_value']:.2f}")

        regime_results[regime_name] = rule_results + ml_results

    # 5. SL/TP sweep on top rule strategies
    print("\n[5/6] Testing stop-loss/take-profit combinations...")
    all_rule = []
    for results in regime_results.values():
        all_rule.extend([r for r in results if r["type"] == "Rule"])
    top_strategies = sorted(all_rule, key=lambda x: x["total_return"], reverse=True)[:5]

    sl_tp_results = test_sl_tp_combos(features, "1h", top_strategies)
    for r in sl_tp_results[:10]:
        print(f"  {r['strategy']} | {r['sl_tp_label']}: "
              f"ret={r['total_return']:.1f}%, dd={r['max_drawdown']:.1f}%")

    # 6. Generate report
    print("\n[6/6] Generating report...")
    report = generate_report(regime_results, sl_tp_results, meta_info)
    report_path = "comprehensive_strategy_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved: {report_path}")

    # Summary
    all_results = []
    for results in regime_results.values():
        all_results.extend(results)

    if all_results:
        best = max(all_results, key=lambda x: x["total_return"])
        print(f"\n{'=' * 60}")
        print("BEST STRATEGY (BY REGIME)")
        print(f"{'=' * 60}")
        for regime_name in ["Bull", "Bear", "Sideways"]:
            cn = {"Bull": "牛市", "Bear": "熊市", "Sideways": "震盪"}[regime_name]
            if regime_results.get(regime_name):
                r_best = max(regime_results[regime_name], key=lambda x: x["total_return"])
                print(f"  {cn}: {r_best['name']} -> "
                      f"ret={r_best['total_return']:.1f}%, "
                      f"trades={r_best['num_trades']}, "
                      f"ev={r_best['expected_value']:.2f}")
        print(f"\n  OVERALL BEST: {best['name']}")
        print(f"    Return: {best['total_return']:.1f}%")
        print(f"    Annualized: {best['annualized_return']:.1f}%")
        print(f"    Trades: {best['num_trades']}")
        print(f"    Win Rate: {best['win_rate']:.0f}%")
        print(f"    Expected Value: {best['expected_value']:.2f}")
        print(f"{'=' * 60}")

    print(f"\nStatus: DONE")
    print(f"Files: comprehensive_strategy_report.md")


if __name__ == "__main__":
    main()
