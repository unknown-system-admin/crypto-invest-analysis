#!/usr/bin/env python
"""Comprehensive visual report generator for crypto investment analysis.

Fetches 2 years of hourly data and generates 5 visual reports:
1. Data Overview (price, volume, statistics)
2. Technical Indicators (SMA, RSI, MACD, OBV)
3. Strategy Performance (equity curves, drawdown, monthly returns)
4. Model Performance (learning curves, feature importance, confusion matrix)
5. Comparison Report (strategy comparison, risk-return, Sharpe ratios)
"""

import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import ta
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from data_cache import load_or_fetch, save_to_cache, CACHE_DIR
from feature_engine.indicators import compute_all_indicators
from feature_engine.momentum import momentum_score, momentum_delta
from backtest_engine.engine import BacktestEngine
from backtest_engine.rule_strategy import MomentumRuleStrategy
from backtest_engine.short_strategy import MomentumShortStrategy
from backtest_engine.strategy import Signal

warnings.filterwarnings("ignore")

# ── Constants ──────────────────────────────────────────────────────────

REPORTS_DIR = Path(__file__).parent / "reports"
SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
TWO_YEARS_CANDLES = 17520  # ~2 years of hourly data
FETCH_BATCH = 300

plt.style.use("seaborn-v0_8-darkgrid")
COLORS = {
    "blue": "#2196F3",
    "red": "#F44336",
    "green": "#4CAF50",
    "orange": "#FF9800",
    "purple": "#9C27B0",
    "cyan": "#00BCD4",
    "pink": "#E91E63",
    "teal": "#009688",
}


# ── Data Fetching ──────────────────────────────────────────────────────


def fetch_2years_data(force_refresh: bool = False) -> pd.DataFrame:
    """Fetch 2 years of hourly data with batch caching."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / "BTC_USDT_1h.csv"

    if not force_refresh and cache_file.exists():
        df = pd.read_csv(cache_file, index_col="timestamp", parse_dates=True)
        if len(df) >= TWO_YEARS_CANDLES:
            print(f"Cache sufficient: {len(df)} candles ({df.index[0]} -> {df.index[-1]})")
            return df
        print(f"Cache has {len(df)} candles, need {TWO_YEARS_CANDLES}. Fetching more...")

    print(f"Fetching {TWO_YEARS_CANDLES} candles for {SYMBOL} {TIMEFRAME}...")
    df = load_or_fetch(SYMBOL, TIMEFRAME, limit=TWO_YEARS_CANDLES, force_refresh=force_refresh)

    if len(df) < TWO_YEARS_CANDLES:
        print(f"Warning: Got {len(df)} candles (target: {TWO_YEARS_CANDLES})")

    print(f"Data ready: {len(df)} candles ({df.index[0]} -> {df.index[-1]})")
    return df


# ── Feature Building ──────────────────────────────────────────────────


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build full feature set for model training."""
    indicators = compute_all_indicators(df)
    indicators["close"] = df["close"]
    indicators["momentum_score"] = momentum_score(indicators)
    indicators["momentum_delta"] = momentum_delta(indicators["momentum_score"])
    return indicators


def generate_labels(df: pd.DataFrame, n_bars: int = 5) -> pd.Series:
    """Binary labels: 1 = price goes up in next n_bars, 0 = down."""
    future = df["close"].shift(-n_bars)
    ret = ((future - df["close"]) / df["close"]) * 100
    labels = (ret > 0).astype(float)
    labels[ret.isna()] = np.nan
    return labels


# ── Report A: Data Overview ───────────────────────────────────────────


def generate_data_overview(df: pd.DataFrame) -> None:
    """Generate data overview report: price chart, volume, statistics."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), gridspec_kw={"height_ratios": [3, 1, 1]})
    fig.suptitle(f"Data Overview — {SYMBOL} {TIMEFRAME}", fontsize=16, fontweight="bold", y=0.98)

    # Price chart
    ax_price = axes[0]
    ax_price.plot(df.index, df["close"], color=COLORS["blue"], linewidth=0.8, label="Close Price")
    ax_price.fill_between(df.index, df["close"], alpha=0.1, color=COLORS["blue"])
    ax_price.set_ylabel("Price (USDT)", fontsize=11)
    ax_price.legend(loc="upper left", fontsize=10)
    ax_price.set_title("Price Chart", fontsize=12, pad=10)

    # Volume chart
    ax_vol = axes[1]
    vol_colors = [COLORS["green"] if df["close"].iloc[i] >= df["open"].iloc[i] else COLORS["red"]
                  for i in range(len(df))]
    ax_vol.bar(df.index, df["volume"], color=vol_colors, alpha=0.7, width=0.02)
    ax_vol.set_ylabel("Volume", fontsize=11)
    ax_vol.set_title("Volume", fontsize=12, pad=10)

    # Statistics table
    ax_stats = axes[2]
    ax_stats.axis("off")
    stats = {
        "Total Candles": f"{len(df):,}",
        "Date Range": f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}",
        "Days Covered": f"{(df.index[-1] - df.index[0]).days}",
        "Start Price": f"${df['close'].iloc[0]:,.2f}",
        "End Price": f"${df['close'].iloc[-1]:,.2f}",
        "Total Return": f"{((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]) * 100:.2f}%",
        "Max Price": f"${df['high'].max():,.2f}",
        "Min Price": f"${df['low'].min():,.2f}",
        "Avg Volume": f"{df['volume'].mean():,.0f}",
        "Volatility (std)": f"{df['close'].pct_change().std() * 100:.2f}%",
    }

    table_data = [[k, v] for k, v in stats.items()]
    table = ax_stats.table(
        cellText=table_data,
        colLabels=["Metric", "Value"],
        cellLoc="center",
        loc="center",
        colWidths=[0.3, 0.3],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor(COLORS["blue"])
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f0f0f0")

    for ax in axes[:2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = REPORTS_DIR / "01_data_overview.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Report B: Technical Indicators ────────────────────────────────────


def generate_indicators_report(df: pd.DataFrame) -> None:
    """Generate technical indicators report: SMA, RSI, MACD, OBV."""
    indicators = compute_all_indicators(df)

    fig, axes = plt.subplots(4, 1, figsize=(16, 14), gridspec_kw={"height_ratios": [3, 1, 1, 1]})
    fig.suptitle(f"Technical Indicators — {SYMBOL} {TIMEFRAME}", fontsize=16, fontweight="bold", y=0.98)

    # Price with SMAs
    ax1 = axes[0]
    ax1.plot(df.index, df["close"], color=COLORS["blue"], linewidth=0.8, label="Close")
    ax1.plot(indicators.index, indicators["SMA_20"], color=COLORS["orange"], linewidth=1, label="SMA 20")
    ax1.plot(indicators.index, indicators["SMA_50"], color=COLORS["green"], linewidth=1, label="SMA 50")
    ax1.set_ylabel("Price (USDT)", fontsize=11)
    ax1.legend(loc="upper left", fontsize=10)
    ax1.set_title("Price with Moving Averages", fontsize=12, pad=10)

    # RSI
    ax2 = axes[1]
    ax2.plot(indicators.index, indicators["RSI"], color=COLORS["purple"], linewidth=0.8)
    ax2.axhline(y=70, color=COLORS["red"], linestyle="--", alpha=0.7, label="Overbought (70)")
    ax2.axhline(y=30, color=COLORS["green"], linestyle="--", alpha=0.7, label="Oversold (30)")
    ax2.fill_between(indicators.index, 70, 100, alpha=0.1, color=COLORS["red"])
    ax2.fill_between(indicators.index, 0, 30, alpha=0.1, color=COLORS["green"])
    ax2.set_ylabel("RSI", fontsize=11)
    ax2.set_ylim(0, 100)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.set_title("RSI (14)", fontsize=12, pad=10)

    # MACD
    ax3 = axes[2]
    macd_line = indicators["MACD"]
    signal_line = ta.trend.MACD(df["close"], window_slow=26, window_fast=12, window_sign=9).macd_signal()
    histogram = macd_line - signal_line

    ax3.plot(indicators.index, macd_line, color=COLORS["blue"], linewidth=0.8, label="MACD")
    ax3.plot(indicators.index, signal_line, color=COLORS["orange"], linewidth=0.8, label="Signal")
    hist_colors = [COLORS["green"] if v >= 0 else COLORS["red"] for v in histogram]
    ax3.bar(indicators.index, histogram, color=hist_colors, alpha=0.5, width=0.02, label="Histogram")
    ax3.axhline(y=0, color="gray", linewidth=0.5)
    ax3.set_ylabel("MACD", fontsize=11)
    ax3.legend(loc="upper left", fontsize=9)
    ax3.set_title("MACD (12, 26, 9)", fontsize=12, pad=10)

    # OBV
    ax4 = axes[3]
    ax4.plot(indicators.index, indicators["OBV"], color=COLORS["teal"], linewidth=0.8)
    ax4.set_ylabel("OBV", fontsize=11)
    ax4.set_title("On Balance Volume", fontsize=12, pad=10)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = REPORTS_DIR / "02_indicators.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Report C: Strategy Performance ────────────────────────────────────


def generate_strategy_performance(df: pd.DataFrame) -> None:
    """Generate strategy performance report: equity curves, drawdown, monthly returns."""
    features = build_features(df)
    features = features.dropna()

    # Run strategies
    strategies = {
        "Momentum Rule": MomentumRuleStrategy(buy_threshold=0.08, sell_threshold=-0.07),
        "Momentum Short": MomentumShortStrategy(buy_threshold=0.01, sell_threshold=-0.01),
    }

    results = {}
    for name, strategy in strategies.items():
        engine = BacktestEngine(strategy=strategy, initial_capital=10000, timeframe=TIMEFRAME)
        result = engine.run(features)
        results[name] = result

    fig, axes = plt.subplots(3, 1, figsize=(16, 14), gridspec_kw={"height_ratios": [2, 1, 1.5]})
    fig.suptitle("Strategy Performance", fontsize=16, fontweight="bold", y=0.98)

    # Equity curves
    ax1 = axes[0]
    for i, (name, result) in enumerate(results.items()):
        color = list(COLORS.values())[i]
        ax1.plot(range(len(result.equity_curve)), result.equity_curve,
                 color=color, linewidth=1, label=f"{name} ({result.total_return_pct:+.1f}%)")
    ax1.axhline(y=10000, color="gray", linestyle="--", alpha=0.5, label="Initial Capital")
    ax1.set_ylabel("Equity (USDT)", fontsize=11)
    ax1.legend(loc="upper left", fontsize=10)
    ax1.set_title("Equity Curves", fontsize=12, pad=10)

    # Drawdown chart
    ax2 = axes[1]
    for i, (name, result) in enumerate(results.items()):
        equity = np.array(result.equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak * 100
        color = list(COLORS.values())[i]
        ax2.fill_between(range(len(drawdown)), drawdown, alpha=0.3, color=color, label=name)
        ax2.plot(range(len(drawdown)), drawdown, color=color, linewidth=0.5)
    ax2.set_ylabel("Drawdown (%)", fontsize=11)
    ax2.invert_yaxis()
    ax2.legend(loc="lower left", fontsize=10)
    ax2.set_title("Drawdown", fontsize=12, pad=10)

    # Monthly returns heatmap (first strategy)
    ax3 = axes[2]
    first_result = list(results.values())[0]
    equity_series = pd.Series(first_result.equity_curve, index=features.index[:len(first_result.equity_curve)])
    monthly = equity_series.resample("ME").last().pct_change() * 100

    if len(monthly) > 1:
        monthly_pivot = pd.DataFrame({
            "Year": monthly.index.year,
            "Month": monthly.index.month,
            "Return": monthly.values,
        })
        pivot = monthly_pivot.pivot_table(values="Return", index="Year", columns="Month", aggfunc="first")
        pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][:len(pivot.columns)]

        im = ax3.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=-10, vmax=10)
        ax3.set_xticks(range(len(pivot.columns)))
        ax3.set_xticklabels(pivot.columns, fontsize=10)
        ax3.set_yticks(range(len(pivot.index)))
        ax3.set_yticklabels(pivot.index, fontsize=10)

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax3.text(j, i, f"{val:.1f}%", ha="center", va="center", fontsize=8,
                             color="white" if abs(val) > 5 else "black")

        plt.colorbar(im, ax=ax3, label="Monthly Return (%)", shrink=0.8)
    ax3.set_title("Monthly Returns Heatmap (Momentum Rule)", fontsize=12, pad=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = REPORTS_DIR / "03_strategy_performance.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Report D: Model Performance ───────────────────────────────────────


def generate_model_performance(df: pd.DataFrame) -> None:
    """Generate model performance report: learning curves, feature importance, confusion matrix."""
    features = build_features(df)
    labels = generate_labels(df)

    valid = features.dropna().index.intersection(labels.dropna().index)
    X = features.loc[valid]
    y = labels.loc[valid]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Train model with incremental learning for curve
    model = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42, n_jobs=-1)

    train_scores = []
    test_scores = []
    n_steps = 20
    step_size = max(1, len(X_train_s) // n_steps)

    for i in range(step_size, len(X_train_s) + 1, step_size):
        model_clone = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42, n_jobs=-1)
        model_clone.fit(X_train_s[:i], y_train.iloc[:i])
        train_scores.append(accuracy_score(y_train.iloc[:i], model_clone.predict(X_train_s[:i])))
        test_scores.append(accuracy_score(y_test, model_clone.predict(X_test_s)))

    # Final model
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Model Performance (Random Forest)", fontsize=16, fontweight="bold", y=0.98)

    # Learning curves
    ax1 = axes[0, 0]
    steps = range(1, len(train_scores) + 1)
    ax1.plot(steps, train_scores, color=COLORS["blue"], linewidth=1.5, label="Train Accuracy", marker="o", markersize=3)
    ax1.plot(steps, test_scores, color=COLORS["red"], linewidth=1.5, label="Test Accuracy", marker="s", markersize=3)
    ax1.set_xlabel("Training Steps", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.set_title("Learning Curves", fontsize=12, pad=10)
    ax1.grid(True, alpha=0.3)

    # Feature importance
    ax2 = axes[0, 1]
    importances = model.feature_importances_
    fi = sorted(zip(X.columns, importances), key=lambda x: x[1], reverse=True)
    names, vals = zip(*fi)
    colors_bar = [COLORS["blue"] if v > np.median(vals) else COLORS["orange"] for v in vals]
    bars = ax2.barh(range(len(names)), vals, color=colors_bar, alpha=0.8)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, fontsize=9)
    ax2.set_xlabel("Importance", fontsize=11)
    ax2.set_title("Feature Importance", fontsize=12, pad=10)
    ax2.invert_yaxis()

    # Confusion matrix
    ax3 = axes[1, 0]
    cm = confusion_matrix(y_test, y_pred)
    im = ax3.imshow(cm, cmap="Blues", aspect="auto")
    ax3.set_xticks([0, 1])
    ax3.set_yticks([0, 1])
    ax3.set_xticklabels(["Down", "Up"], fontsize=10)
    ax3.set_yticklabels(["Down", "Up"], fontsize=10)
    ax3.set_xlabel("Predicted", fontsize=11)
    ax3.set_ylabel("Actual", fontsize=11)
    for i in range(2):
        for j in range(2):
            ax3.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14,
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im, ax=ax3, shrink=0.8)
    ax3.set_title("Confusion Matrix", fontsize=12, pad=10)

    # Metrics summary
    ax4 = axes[1, 1]
    ax4.axis("off")
    metrics = {
        "Train Accuracy": f"{accuracy_score(y_train, model.predict(X_train_s)):.2%}",
        "Test Accuracy": f"{accuracy_score(y_test, y_pred):.2%}",
        "F1 Score": f"{f1_score(y_test, y_pred, zero_division=0):.4f}",
        "Train Samples": f"{len(X_train):,}",
        "Test Samples": f"{len(X_test):,}",
        "Features": f"{len(X.columns)}",
        "Trees": "100",
        "Max Depth": "7",
    }
    table_data = [[k, v] for k, v in metrics.items()]
    table = ax4.table(
        cellText=table_data,
        colLabels=["Metric", "Value"],
        cellLoc="center",
        loc="center",
        colWidths=[0.4, 0.3],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor(COLORS["purple"])
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f0f0f0")
    ax4.set_title("Model Metrics", fontsize=12, pad=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = REPORTS_DIR / "04_model_performance.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Report E: Comparison Report ───────────────────────────────────────


def generate_comparison_report(df: pd.DataFrame) -> None:
    """Generate comparison report: strategy comparison, risk-return, Sharpe ratios."""
    features = build_features(df)
    features = features.dropna()

    strategies = {
        "Momentum Rule": MomentumRuleStrategy(buy_threshold=0.08, sell_threshold=-0.07),
        "Momentum Short": MomentumShortStrategy(buy_threshold=0.01, sell_threshold=-0.01),
    }

    results = {}
    for name, strategy in strategies.items():
        engine = BacktestEngine(strategy=strategy, initial_capital=10000, timeframe=TIMEFRAME)
        results[name] = engine.run(features)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Strategy Comparison Report", fontsize=16, fontweight="bold", y=0.98)

    # Bar chart comparison
    ax1 = axes[0, 0]
    names = list(results.keys())
    metrics_to_plot = ["total_return_pct", "max_drawdown_pct", "sharpe_ratio", "win_rate"]
    metric_labels = ["Return %", "Max DD %", "Sharpe", "Win Rate %"]
    x = np.arange(len(names))
    width = 0.2

    for i, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        vals = [getattr(results[n], metric) for n in names]
        ax1.bar(x + i * width, vals, width, label=label, alpha=0.8)
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(names, fontsize=10)
    ax1.legend(fontsize=9)
    ax1.set_title("Key Metrics Comparison", fontsize=12, pad=10)
    ax1.grid(True, alpha=0.3, axis="y")

    # Risk-return scatter
    ax2 = axes[0, 1]
    for i, (name, result) in enumerate(results.items()):
        equity = np.array(result.equity_curve)
        returns = np.diff(equity) / equity[:-1]
        annual_vol = returns.std() * np.sqrt(8760) * 100
        annual_ret = result.total_return_pct
        color = list(COLORS.values())[i]
        ax2.scatter(annual_vol, annual_ret, s=200, color=color, zorder=5, edgecolors="black")
        ax2.annotate(name, (annual_vol, annual_ret), textcoords="offset points",
                     xytext=(10, 10), fontsize=10, fontweight="bold")
    ax2.set_xlabel("Annualized Volatility (%)", fontsize=11)
    ax2.set_ylabel("Total Return (%)", fontsize=11)
    ax2.set_title("Risk-Return Scatter", fontsize=12, pad=10)
    ax2.grid(True, alpha=0.3)

    # Sharpe ratio comparison
    ax3 = axes[1, 0]
    sharpe_vals = [results[n].sharpe_ratio for n in names]
    bars = ax3.bar(names, sharpe_vals, color=[COLORS["blue"], COLORS["orange"]], alpha=0.8, edgecolor="black")
    ax3.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    for bar, val in zip(bars, sharpe_vals):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Sharpe Ratio", fontsize=11)
    ax3.set_title("Sharpe Ratio Comparison", fontsize=12, pad=10)
    ax3.grid(True, alpha=0.3, axis="y")

    # Summary table
    ax4 = axes[1, 1]
    ax4.axis("off")
    summary_data = []
    for name, result in results.items():
        summary_data.append([
            name,
            f"{result.total_return_pct:+.1f}%",
            f"{result.max_drawdown_pct:.1f}%",
            f"{result.sharpe_ratio:.2f}",
            f"{result.win_rate:.1f}%",
            str(result.total_trades),
        ])
    table = ax4.table(
        cellText=summary_data,
        colLabels=["Strategy", "Return", "Max DD", "Sharpe", "Win Rate", "Trades"],
        cellLoc="center",
        loc="center",
        colWidths=[0.2, 0.15, 0.15, 0.12, 0.15, 0.12],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor(COLORS["teal"])
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f0f0f0")
    ax4.set_title("Strategy Summary", fontsize=12, pad=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = REPORTS_DIR / "05_comparison.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────


def generate_reports(force_refresh: bool = False) -> None:
    """Generate all visual reports."""
    print("=" * 60)
    print("Crypto Visual Report Generator")
    print("=" * 60)

    # Create reports directory
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Fetch 2 years of data
    print("\n[1/6] Fetching 2 years of data...")
    df = fetch_2years_data(force_refresh=force_refresh)
    print(f"  Data: {len(df)} candles, {df.index[0]} -> {df.index[-1]}")

    # Step 2: Data Overview
    print("\n[2/6] Generating Data Overview report...")
    generate_data_overview(df)

    # Step 3: Technical Indicators
    print("\n[3/6] Generating Technical Indicators report...")
    generate_indicators_report(df)

    # Step 4: Strategy Performance
    print("\n[4/6] Generating Strategy Performance report...")
    generate_strategy_performance(df)

    # Step 5: Model Performance
    print("\n[5/6] Generating Model Performance report...")
    generate_model_performance(df)

    # Step 6: Comparison Report
    print("\n[6/6] Generating Comparison report...")
    generate_comparison_report(df)

    print("\n" + "=" * 60)
    print("All reports generated in reports/ directory")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate visual reports for crypto analysis")
    parser.add_argument("--force-refresh", action="store_true", help="Force re-fetch data from OKX")
    args = parser.parse_args()
    generate_reports(force_refresh=args.force_refresh)
