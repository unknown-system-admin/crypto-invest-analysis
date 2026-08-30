# 指標分析系統 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立完整分析流程，測試動能指標有效性並透過 ML 找出最佳指標

**Architecture:** 建立 `analyzers/` 模組，包含動能分析、特徵篩選、報告生成三個組件

**Tech Stack:** Python, pandas, numpy, scikit-learn, matplotlib, feature_engine, backtest_engine

---

## Task 1: Create analyzers/__init__.py — Package Init

**Files:**
- Create: `analyzers/__init__.py`

- [ ] **Step 1: Create package init**

```python
# analyzers/__init__.py
```

- [ ] **Step 2: Commit**

```bash
git add analyzers/__init__.py
git commit -m "feat: create analyzers package"
```

---

## Task 2: Create analyzers/momentum_analyzer.py — Momentum Effectiveness Analysis

**Files:**
- Create: `analyzers/momentum_analyzer.py`
- Create: `tests/test_momentum_analyzer.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_momentum_analyzer.py
import pandas as pd
import numpy as np
from analyzers.momentum_analyzer import MomentumAnalyzer


def test_momentum_analyzer_correlation():
    dates = pd.date_range("2024-01-01", periods=300, freq="1D")
    np.random.seed(42)
    close = 50000 + np.cumsum(np.random.randn(300) * 500)
    
    df = pd.DataFrame({
        "close": close,
        "momentum_score": np.sin(np.linspace(0, 10, 300)),
    }, index=dates)
    
    analyzer = MomentumAnalyzer()
    result = analyzer.analyze_correlation(df, forward_days=5)
    
    assert "correlation" in result
    assert "p_value" in result
    assert isinstance(result["correlation"], float)


def test_momentum_analyzer_threshold_backtest():
    dates = pd.date_range("2024-01-01", periods=300, freq="1D")
    np.random.seed(42)
    close = 50000 + np.cumsum(np.random.randn(300) * 500)
    
    df = pd.DataFrame({
        "close": close,
        "momentum_score": np.sin(np.linspace(0, 10, 300)),
        "momentum_delta": np.cos(np.linspace(0, 10, 300)),
    }, index=dates)
    
    analyzer = MomentumAnalyzer()
    results = analyzer.backtest_thresholds(
        df,
        buy_thresholds=[0.2, 0.3, 0.4],
        sell_thresholds=[-0.2, -0.3, -0.4],
    )
    
    assert len(results) == 9  # 3x3 combinations
    assert all("buy_threshold" in r for r in results)
    assert all("total_return_pct" in r for r in results)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/unknown965/coding/OpenCodeTest/crypto-invest-analysis && .venv/bin/python -m pytest tests/test_momentum_analyzer.py -v`
Expected: FAIL with "ImportError: cannot import name 'MomentumAnalyzer' from 'analyzers.momentum_analyzer'"

- [ ] **Step 3: Write minimal implementation**

```python
# analyzers/momentum_analyzer.py
import pandas as pd
import numpy as np
from scipy import stats
from feature_engine.momentum import momentum_score, momentum_delta
from feature_engine.labels import future_return
from backtest_engine.rule_strategy import MomentumRuleStrategy
from backtest_engine.engine import BacktestEngine


class MomentumAnalyzer:
    def analyze_correlation(self, df: pd.DataFrame, forward_days: int = 5) -> dict:
        """Analyze correlation between momentum score and future returns."""
        # Compute momentum if not present
        if "momentum_score" not in df.columns:
            indicators = df[["close"]].copy()
            indicators["RSI"] = 50  # placeholder
            indicators["MACD_histogram"] = 0  # placeholder
            indicators["SMA_20"] = df["close"]
            indicators["SMA_50"] = df["close"]
            df["momentum_score"] = momentum_score(indicators)
        
        # Calculate future returns
        future_ret = future_return(df, n_bars=forward_days)
        
        # Remove NaN
        valid_idx = df["momentum_score"].dropna().index.intersection(future_ret.dropna().index)
        x = df.loc[valid_idx, "momentum_score"]
        y = future_ret.loc[valid_idx]
        
        # Calculate correlation
        corr, p_value = stats.pearsonr(x, y)
        
        return {
            "correlation": round(corr, 4),
            "p_value": round(p_value, 4),
            "significant": p_value < 0.05,
        }
    
    def backtest_thresholds(
        self,
        df: pd.DataFrame,
        buy_thresholds: list,
        sell_thresholds: list,
    ) -> list:
        """Backtest different threshold combinations."""
        results = []
        
        for buy_thresh in buy_thresholds:
            for sell_thresh in sell_thresholds:
                strategy = MomentumRuleStrategy(
                    buy_threshold=buy_thresh,
                    sell_threshold=sell_thresh,
                )
                engine = BacktestEngine(
                    strategy=strategy,
                    initial_capital=10000,
                    symbol="BTC/USDT",
                    timeframe="1d",
                )
                
                # Build features
                features = df[["close", "momentum_score", "momentum_delta"]].copy()
                features = features.dropna()
                
                result = engine.run(features)
                
                results.append({
                    "buy_threshold": buy_thresh,
                    "sell_threshold": sell_thresh,
                    "total_trades": result.total_trades,
                    "total_return_pct": result.total_return_pct,
                    "max_drawdown_pct": result.max_drawdown_pct,
                    "sharpe_ratio": result.sharpe_ratio,
                    "win_rate": result.win_rate,
                })
        
        return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/unknown965/coding/OpenCodeTest/crypto-invest-analysis && .venv/bin/python -m pytest tests/test_momentum_analyzer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add analyzers/momentum_analyzer.py tests/test_momentum_analyzer.py
git commit -m "feat: add analyzers/momentum_analyzer.py for momentum analysis"
```

---

## Task 3: Create analyzers/feature_selector.py — ML Feature Selection

**Files:**
- Create: `analyzers/feature_selector.py`
- Create: `tests/test_feature_selector.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_feature_selector.py
import pandas as pd
import numpy as np
from analyzers.feature_selector import FeatureSelector


def test_feature_selector_train():
    dates = pd.date_range("2024-01-01", periods=300, freq="1D")
    np.random.seed(42)
    
    df = pd.DataFrame({
        "close": 50000 + np.cumsum(np.random.randn(300) * 500),
        "RSI": np.random.uniform(30, 70, 300),
        "MACD": np.random.randn(300) * 100,
        "SMA_20": np.random.uniform(40000, 50000, 300),
        "SMA_50": np.random.uniform(40000, 50000, 300),
    }, index=dates)
    
    # Add binary label
    df["binary_label"] = (df["close"].shift(-5) > df["close"]).astype(int)
    df = df.dropna()
    
    selector = FeatureSelector()
    result = selector.train(
        df,
        feature_columns=["RSI", "MACD", "SMA_20", "SMA_50"],
        label_column="binary_label",
    )
    
    assert "accuracy" in result
    assert "feature_importance" in result
    assert len(result["feature_importance"]) == 4


def test_feature_selector_get_top_features():
    selector = FeatureSelector()
    selector.feature_importance = {
        "RSI": 0.3,
        "MACD": 0.25,
        "SMA_20": 0.2,
        "SMA_50": 0.15,
        "ATR": 0.1,
    }
    
    top = selector.get_top_features(n=3)
    
    assert len(top) == 3
    assert top[0] == "RSI"
    assert top[1] == "MACD"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/unknown965/coding/OpenCodeTest/crypto-invest-analysis && .venv/bin/python -m pytest tests/test_feature_selector.py -v`
Expected: FAIL with "ImportError: cannot import name 'FeatureSelector' from 'analyzers.feature_selector'"

- [ ] **Step 3: Write minimal implementation**

```python
# analyzers/feature_selector.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class FeatureSelector:
    def __init__(self):
        self.model = None
        self.feature_importance = {}
    
    def train(
        self,
        df: pd.DataFrame,
        feature_columns: list,
        label_column: str,
    ) -> dict:
        """Train model and calculate feature importance."""
        X = df[feature_columns]
        y = df[label_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Feature importance
        self.feature_importance = dict(zip(
            feature_columns,
            self.model.feature_importances_
        ))
        
        return {
            "accuracy": round(accuracy, 4),
            "feature_importance": self.feature_importance,
        }
    
    def get_top_features(self, n: int = 10) -> list:
        """Get top N features by importance."""
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return [f[0] for f in sorted_features[:n]]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/unknown965/coding/OpenCodeTest/crypto-invest-analysis && .venv/bin/python -m pytest tests/test_feature_selector.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add analyzers/feature_selector.py tests/test_feature_selector.py
git commit -m "feat: add analyzers/feature_selector.py for ML feature selection"
```

---

## Task 4: Create analyzers/report_generator.py — Report Generation

**Files:**
- Create: `analyzers/report_generator.py`
- Create: `tests/test_report_generator.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_report_generator.py
from analyzers.report_generator import ReportGenerator


def test_report_generator_momentum_section():
    generator = ReportGenerator()
    
    correlation_result = {
        "correlation": 0.25,
        "p_value": 0.01,
        "significant": True,
    }
    
    section = generator.momentum_section(correlation_result)
    
    assert "0.25" in section
    assert "顯著" in section


def test_report_generator_feature_section():
    generator = ReportGenerator()
    
    feature_result = {
        "accuracy": 0.65,
        "feature_importance": {
            "RSI": 0.3,
            "MACD": 0.25,
            "SMA_20": 0.2,
        },
    }
    
    section = generator.feature_section(feature_result)
    
    assert "0.65" in section
    assert "RSI" in section
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/unknown965/coding/OpenCodeTest/crypto-invest-analysis && .venv/bin/python -m pytest tests/test_report_generator.py -v`
Expected: FAIL with "ImportError: cannot import name 'ReportGenerator' from 'analyzers.report_generator'"

- [ ] **Step 3: Write minimal implementation**

```python
# analyzers/report_generator.py
class ReportGenerator:
    def momentum_section(self, correlation_result: dict) -> str:
        """Generate momentum analysis section."""
        corr = correlation_result["correlation"]
        significant = correlation_result["significant"]
        
        sig_text = "顯著" if significant else "不顯著"
        
        return f"""
## 動能指標有效性分析

| 指標 | 數值 |
|------|------|
| 相關係數 | {corr} |
| 統計顯著性 | {sig_text} |

結論：動能指標與未來報酬的相關性為 {sig_text}。
"""
    
    def feature_section(self, feature_result: dict) -> str:
        """Generate feature importance section."""
        accuracy = feature_result["accuracy"]
        importance = feature_result["feature_importance"]
        
        # Sort by importance
        sorted_features = sorted(
            importance.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        
        lines = ["## ML 特徵重要性分析\n"]
        lines.append(f"模型準確率：{accuracy}\n")
        lines.append("| 排名 | 指標 | 重要性分數 |")
        lines.append("|------|------|-----------|")
        
        for i, (feature, score) in enumerate(sorted_features[:10], 1):
            lines.append(f"| {i} | {feature} | {score:.4f} |")
        
        return "\n".join(lines)
    
    def full_report(
        self,
        momentum_result: dict,
        feature_result: dict,
        backtest_results: list,
    ) -> str:
        """Generate full analysis report."""
        sections = [
            "# 指標分析報告\n",
            self.momentum_section(momentum_result),
            self.feature_section(feature_result),
            self.backtest_section(backtest_results),
        ]
        
        return "\n".join(sections)
    
    def backtest_section(self, backtest_results: list) -> str:
        """Generate backtest results section."""
        if not backtest_results:
            return "## 回測結果\n\n無回測數據"
        
        # Find best result by Sharpe ratio
        best = max(backtest_results, key=lambda x: x.get("sharpe_ratio", 0))
        
        lines = ["## 回測結果\n"]
        lines.append("### 最佳參數組合\n")
        lines.append(f"- Buy Threshold: {best['buy_threshold']}")
        lines.append(f"- Sell Threshold: {best['sell_threshold']}")
        lines.append(f"- 報酬率: {best['total_return_pct']:.1f}%")
        lines.append(f"- 最大回撤: {best['max_drawdown_pct']:.1f}%")
        lines.append(f"- 夏普比率: {best['sharpe_ratio']:.2f}")
        lines.append(f"- 勝率: {best['win_rate']:.1f}%")
        
        return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/unknown965/coding/OpenCodeTest/crypto-invest-analysis && .venv/bin/python -m pytest tests/test_report_generator.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add analyzers/report_generator.py tests/test_report_generator.py
git commit -m "feat: add analyzers/report_generator.py for report generation"
```

---

## Task 5: Create run_analysis.py — Main Analysis Script

**Files:**
- Create: `run_analysis.py`

- [ ] **Step 1: Create the script**

```python
#!/usr/bin/env python3
"""Main analysis script for indicator effectiveness and ML feature selection."""
import sys
sys.path.insert(0, ".")

import ccxt
import pandas as pd
from datetime import datetime, timedelta
from feature_engine.builder import build_feature_matrix
from feature_engine.momentum import momentum_score, momentum_delta
from analyzers.momentum_analyzer import MomentumAnalyzer
from analyzers.feature_selector import FeatureSelector
from analyzers.report_generator import ReportGenerator


def main():
    print("=" * 60)
    print("指標分析系統")
    print("=" * 60)
    
    # 1. Fetch data
    print("\n[1/5] 從 OKX 獲取數據...")
    exchange = ccxt.okx()
    since = int((datetime.now() - timedelta(days=365)).timestamp() * 1000)
    ohlcv = exchange.fetch_ohlcv("BTC/USDT", "1d", since=since, limit=365)
    
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    print(f"  數據範圍: {df.index[0].date()} 至 {df.index[-1].date()}")
    print(f"  數據筆數: {len(df)}")
    
    # 2. Build features
    print("\n[2/5] 計算技術指標...")
    features, labels = build_feature_matrix(df, n_bars=5)
    print(f"  特徵數量: {features.shape[1]}")
    print(f"  有效樣本: {features.shape[0]}")
    
    # 3. Momentum analysis
    print("\n[3/5] 動能指標有效性分析...")
    momentum_analyzer = MomentumAnalyzer()
    
    # Add momentum features to df for analysis
    df["momentum_score"] = features["momentum_score"]
    df["momentum_delta"] = features["momentum_delta"]
    
    correlation_result = momentum_analyzer.analyze_correlation(df, forward_days=5)
    print(f"  相關係數: {correlation_result['correlation']}")
    print(f"  P 值: {correlation_result['p_value']}")
    print(f"  顯著性: {'顯著' if correlation_result['significant'] else '不顯著'}")
    
    # 4. Threshold backtest
    print("\n[4/5] 閾值回測...")
    backtest_results = momentum_analyzer.backtest_thresholds(
        df,
        buy_thresholds=[0.2, 0.3, 0.4, 0.5],
        sell_thresholds=[-0.2, -0.3, -0.4, -0.5],
    )
    
    # Find best result
    best = max(backtest_results, key=lambda x: x.get("sharpe_ratio", 0))
    print(f"  最佳閾值: buy={best['buy_threshold']}, sell={best['sell_threshold']}")
    print(f"  報酬率: {best['total_return_pct']:.1f}%")
    print(f"  夏普比率: {best['sharpe_ratio']:.2f}")
    
    # 5. ML feature selection
    print("\n[5/5] ML 特徵篩選...")
    selector = FeatureSelector()
    
    # Prepare data for ML
    ml_features = features.drop(columns=["close"], errors="ignore")
    ml_features["binary_label"] = labels
    
    # Remove NaN
    ml_features = ml_features.dropna()
    
    feature_columns = [col for col in ml_features.columns if col != "binary_label"]
    
    ml_result = selector.train(
        ml_features,
        feature_columns=feature_columns,
        label_column="binary_label",
    )
    
    top_features = selector.get_top_features(n=10)
    print(f"  模型準確率: {ml_result['accuracy']:.4f}")
    print(f"  Top 10 指標: {', '.join(top_features)}")
    
    # Generate report
    print("\n生成分析報告...")
    generator = ReportGenerator()
    report = generator.full_report(
        momentum_result=correlation_result,
        feature_result=ml_result,
        backtest_results=backtest_results,
    )
    
    # Save report
    report_path = "analysis_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"\n報告已儲存至: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Make script executable**

```bash
chmod +x run_analysis.py
```

- [ ] **Step 3: Commit**

```bash
git add run_analysis.py
git commit -m "feat: add run_analysis.py main analysis script"
```

---

## Task 6: Run Analysis and Verify

- [ ] **Step 1: Run the analysis script**

Run: `cd /Users/unknown965/coding/OpenCodeTest/crypto-invest-analysis && .venv/bin/python run_analysis.py`
Expected: Complete analysis with report generated

- [ ] **Step 2: Run all tests**

Run: `cd /Users/unknown965/coding/OpenCodeTest/crypto-invest-analysis && .venv/bin/python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "feat: complete indicator analysis system"
```

---

## Summary

| Task | Module | Description |
|------|--------|-------------|
| 1 | analyzers/__init__.py | Package init |
| 2 | analyzers/momentum_analyzer.py | Momentum effectiveness analysis |
| 3 | analyzers/feature_selector.py | ML feature selection |
| 4 | analyzers/report_generator.py | Report generation |
| 5 | run_analysis.py | Main analysis script |
| 6 | — | Integration testing |
