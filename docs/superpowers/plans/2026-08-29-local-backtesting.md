# Local Backtesting System Implementation Plan ✅ 已完成

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build unified feature engine + dual-mode backtesting system with momentum quantification and ML pipeline support.

**Architecture:** Create `feature_engine/` module for indicator computation and momentum quantification, `backtest_engine/` for rule-based and model-based backtesting, and `colab/` for ML pipeline notebook.

**Tech Stack:** Python, pandas, numpy, ta (technical analysis library), scikit-learn (for ML), pytest

---

## Task 1: Create feature_engine/indicators.py — Compute 18 Technical Indicators

**Files:**
- Create: `feature_engine/__init__.py`
- Create: `feature_engine/indicators.py`
- Create: `tests/test_feature_engine_indicators.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_feature_engine_indicators.py
import pandas as pd
import numpy as np
from feature_engine.indicators import compute_all_indicators


def test_compute_all_indicators_returns_expected_columns():
    dates = pd.date_range("2024-01-01", periods=100, freq="1h")
    df = pd.DataFrame({
        "open": np.random.uniform(40000, 50000, 100),
        "high": np.random.uniform(40000, 50000, 100),
        "low": np.random.uniform(40000, 50000, 100),
        "close": np.random.uniform(40000, 50000, 100),
        "volume": np.random.uniform(100, 1000, 100),
    }, index=dates)
    
    result = compute_all_indicators(df)
    
    expected_columns = [
        "SMA_20", "SMA_50", "SMA_200", "EMA_12", "EMA_26",
        "ADX", "ICHIMOKU_A", "ICHIMOKU_B",
        "BB_upper", "BB_middle", "BB_lower", "ATR", "KC_upper", "KC_lower",
        "RSI", "MACD", "MACD_signal", "MACD_histogram",
        "STOCH_K", "STOCH_D", "CCI", "Williams_R", "ROC", "MFI",
        "OBV", "VWAP", "CMF",
    ]
    for col in expected_columns:
        assert col in result.columns, f"Missing column: {col}"


def test_compute_all_indicators Handles_nan_gracefully():
    dates = pd.date_range("2024-01-01", periods=50, freq="1h")
    df = pd.DataFrame({
        "open": np.random.uniform(40000, 50000, 50),
        "high": np.random.uniform(40000, 50000, 50),
        "low": np.random.uniform(40000, 50000, 50),
        "close": np.random.uniform(40000, 50000, 50),
        "volume": np.random.uniform(100, 1000, 50),
    }, index=dates)
    
    result = compute_all_indicators(df)
    assert not result.empty
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/unknown965/coding/OpenCodeTest/crypto-invest-analysis && .venv/bin/python -m pytest tests/test_feature_engine_indicators.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'feature_engine'"

- [ ] **Step 3: Create feature_engine/__init__.py**

```python
# feature_engine/__init__.py
```

- [ ] **Step 4: Write minimal implementation**

```python
# feature_engine/indicators.py
import pandas as pd
import ta


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all 18 technical indicators.
    
    Args:
        df: OHLCV DataFrame with columns: open, high, low, close, volume
        
    Returns:
        DataFrame with all indicator columns
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    
    # Trend indicators
    sma_20 = ta.trend.sma_indicator(close, window=20)
    sma_50 = ta.trend.sma_indicator(close, window=50)
    sma_200 = ta.trend.sma_indicator(close, window=200)
    ema_12 = ta.trend.ema_indicator(close, window=12)
    ema_26 = ta.trend.ema_indicator(close, window=26)
    adx = ta.trend.adx(high, low, close, window=14)
    
    # Ichimoku Cloud
    ichimoku = ta.trend.IchimokuIndicator(high, low, window1=9, window2=26, window3=52)
    ichimoku_a = ichimoku.ichimoku_a()
    ichimoku_b = ichimoku.ichimoku_b()
    
    # Volatility indicators
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    bb_upper = bb.bollinger_hband()
    bb_middle = bb.bollinger_mavg()
    bb_lower = bb.bollinger_lband()
    atr = ta.volatility.average_true_range(high, low, close, window=14)
    
    # Keltner Channels
    kc = ta.volatility.KeltnerChannel(high, low, close, window=20, window_atr=20)
    kc_upper = kc.keltner_channel_hband()
    kc_lower = kc.keltner_channel_lband()
    
    # Momentum indicators
    rsi = ta.momentum.rsi(close, window=14)
    macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    macd_line = macd.macd()
    macd_signal = macd.macd_signal()
    macd_histogram = macd.macd_diff()
    
    stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
    stoch_k = stoch.stoch()
    stoch_d = stoch.stoch_signal()
    
    cci = ta.trend.cci(high, low, close, window=20)
    williams_r = ta.momentum.williams_r(high, low, close, lbp=14)
    roc = ta.momentum.roc(close, window=12)
    mfi = ta.volume.money_flow_index(high, low, close, volume, window=14)
    
    # Volume indicators
    obv = ta.volume.on_balance_volume(close, volume)
    vwap = (volume * (high + low + close) / 3).cumsum() / volume.cumsum()
    cmf = ta.volume.chaikin_money_flow(high, low, close, volume, window=20)
    
    result = pd.DataFrame({
        "SMA_20": sma_20,
        "SMA_50": sma_50,
        "SMA_200": sma_200,
        "EMA_12": ema_12,
        "EMA_26": ema_26,
        "ADX": adx,
        "ICHIMOKU_A": ichimoku_a,
        "ICHIMOKU_B": ichimoku_b,
        "BB_upper": bb_upper,
        "BB_middle": bb_middle,
        "BB_lower": bb_lower,
        "ATR": atr,
        "KC_upper": kc_upper,
        "KC_lower": kc_lower,
        "RSI": rsi,
        "MACD": macd_line,
        "MACD_signal": macd_signal,
        "MACD_histogram": macd_histogram,
        "STOCH_K": stoch_k,
        "STOCH_D": stoch_d,
        "CCI": cci,
        "Williams_R": williams_r,
        "ROC": roc,
        "MFI": mfi,
        "OBV": obv,
        "VWAP": vwap,
        "CMF": cmf,
    }, index=df.index)
    
    return result
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /Users/unknown965/coding/OpenCodeTest/crypto-invest-analysis && .venv/bin/python -m pytest tests/test_feature_engine_indicators.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add feature_engine/__init__.py feature_engine/indicators.py tests/test_feature_engine_indicators.py
git commit -m "feat: add feature_engine/indicators.py with 18 technical indicators"
```

---

## Task 2: Create feature_engine/momentum.py — Quantize Momentum to [-1, 1]

**Files:**
- Create: `feature_engine/momentum.py`
- Create: `tests/test_feature_engine_momentum.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_feature_engine_momentum.py
import pandas as pd
import numpy as np
from feature_engine.momentum import momentum_score, momentum_delta, momentum_acceleration


def test_momentum_score_range():
    dates = pd.date_range("2024-01-01", periods=100, freq="1h")
    df = pd.DataFrame({
        "close": np.random.uniform(40000, 50000, 100),
        "RSI": np.random.uniform(30, 70, 100),
        "MACD_histogram": np.random.uniform(-100, 100, 100),
        "SMA_20": np.random.uniform(40000, 50000, 100),
        "SMA_50": np.random.uniform(40000, 50000, 100),
    }, index=dates)
    
    scores = momentum_score(df)
    
    assert scores.min() >= -1.0
    assert scores.max() <= 1.0
    assert len(scores) == 100


def test_momentum_delta_positive_when_increasing():
    scores = pd.Series([0.1, 0.2, 0.4, 0.6, 0.8])
    deltas = momentum_delta(scores)
    
    assert deltas.iloc[1] > 0
    assert deltas.iloc[2] > 0
    assert deltas.iloc[3] > 0


def test_momentum_delta_negative_when_decreasing():
    scores = pd.Series([0.8, 0.6, 0.4, 0.2, 0.1])
    deltas = momentum_delta(scores)
    
    assert deltas.iloc[1] < 0
    assert deltas.iloc[2] < 0
    assert deltas.iloc[3] < 0


def test_momentum_acceleration_positive_when_accelerating():
    scores = pd.Series([0.1, 0.2, 0.4, 0.7, 1.1])
    deltas = momentum_delta(scores)
    accelerations = momentum_acceleration(deltas)
    
    assert accelerations.iloc[2] > 0


def test_momentum_acceleration_negative_when_decelerating():
    scores = pd.Series([0.1, 0.3, 0.5, 0.6, 0.65])
    deltas = momentum_delta(scores)
    accelerations = momentum_acceleration(deltas)
    
    assert accelerations.iloc[2] < 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/unknown965/coding/OpenCodeTest/crypto-invest-analysis && .venv/bin/python -m pytest tests/test_feature_engine_momentum.py -v`
Expected: FAIL with "ImportError: cannot import name 'momentum_score' from 'feature_engine.momentum'"

- [ ] **Step 3: Write minimal implementation**

```python
# feature_engine/momentum.py
import pandas as pd
import numpy as np


def momentum_score(df: pd.DataFrame) -> pd.Series:
    """Calculate momentum score as continuous value [-1, 1].
    
    Args:
        df: DataFrame with columns: close, RSI, MACD_histogram, SMA_20, SMA_50
        
    Returns:
        Series with momentum scores in range [-1, 1]
    """
    # Normalize each component to [-1, 1]
    rsi_norm = (df["RSI"] - 50) / 50
    macd_norm = np.tanh(df["MACD_histogram"] / df["MACD_histogram"].std())
    sma20_norm = (df["close"] - df["SMA_20"]) / df["SMA_20"]
    sma50_norm = (df["close"] - df["SMA_50"]) / df["SMA_50"]
    
    # Weighted combination
    score = (
        0.3 * rsi_norm +
        0.3 * macd_norm +
        0.2 * sma20_norm +
        0.2 * sma50_norm
    )
    
    # Clip to [-1, 1]
    score = score.clip(-1, 1)
    
    return score


def momentum_delta(scores: pd.Series) -> pd.Series:
    """Calculate first derivative of momentum score.
    
    Args:
        scores: Momentum scores
        
    Returns:
        Series with delta values
    """
    return scores.diff()


def momentum_acceleration(deltas: pd.Series) -> pd.Series:
    """Calculate second derivative of momentum score.
    
    Args:
        deltas: Delta values from momentum_delta
        
    Returns:
        Series with acceleration values
    """
    return deltas.diff()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/unknown965/coding/OpenCodeTest/crypto-invest-analysis && .venv/bin/python -m pytest tests/test_feature_engine_momentum.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add feature_engine/momentum.py tests/test_feature_engine_momentum.py
git commit -m "feat: add feature_engine/momentum.py with momentum quantification"
```

---

## Task 3: Create feature_engine/labels.py — Generate Labels for Future Returns

**Files:**
- Create: `feature_engine/labels.py`
- Create: `tests/test_feature_engine_labels.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_feature_engine_labels.py
import pandas as pd
import numpy as np
from feature_engine.labels import future_return, binary_label


def test_future_return_positive():
    dates = pd.date_range("2024-01-01", periods=10, freq="1h")
    df = pd.DataFrame({
        "close": [100, 105, 110, 115, 120, 125, 130, 135, 140, 145],
    }, index=dates)
    
    result = future_return(df, n_bars=2)
    
    assert result.iloc[0] == 10.0  # (110 - 100) / 100 * 100
    assert result.iloc[7] == 5.0   # (145 - 140) / 140 * 100


def test_future_return_negative():
    dates = pd.date_range("2024-01-01", periods=10, freq="1h")
    df = pd.DataFrame({
        "close": [140, 135, 130, 125, 120, 115, 110, 105, 100, 95],
    }, index=dates)
    
    result = future_return(df, n_bars=2)
    
    assert result.iloc[0] < 0


def test_binary_label_up():
    dates = pd.date_range("2024-01-01", periods=10, freq="1h")
    df = pd.DataFrame({
        "close": [100, 105, 110, 115, 120, 125, 130, 135, 140, 145],
    }, index=dates)
    
    result = binary_label(df, n_bars=2, threshold=0)
    
    assert result.iloc[0] == 1  # Up


def test_binary_label_down():
    dates = pd.date_range("2024-01-01", periods=10, freq="1h")
    df = pd.DataFrame({
        "close": [140, 135, 130, 125, 120, 115, 110, 105, 100, 95],
    }, index=dates)
    
    result = binary_label(df, n_bars=2, threshold=0)
    
    assert result.iloc[0] == 0  # Down
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/unknown965/coding/OpenCodeTest/crypto-invest-analysis && .venv/bin/python -m pytest tests/test_feature_engine_labels.py -v`
Expected: FAIL with "ImportError: cannot import name 'future_return' from 'feature_engine.labels'"

- [ ] **Step 3: Write minimal implementation**

```python
# feature_engine/labels.py
import pandas as pd


def future_return(df: pd.DataFrame, n_bars: int = 5) -> pd.Series:
    """Calculate future return percentage.
    
    Args:
        df: DataFrame with 'close' column
        n_bars: Number of bars to look ahead
        
    Returns:
        Series with future return percentages
    """
    future_prices = df["close"].shift(-n_bars)
    return ((future_prices - df["close"]) / df["close"]) * 100


def binary_label(df: pd.DataFrame, n_bars: int = 5, threshold: float = 0) -> pd.Series:
    """Generate binary labels (1=up, 0=down) based on future returns.
    
    Args:
        df: DataFrame with 'close' column
        n_bars: Number of bars to look ahead
        threshold: Minimum return percentage to label as up
        
    Returns:
        Series with binary labels (1 for up, 0 for down)
    """
    returns = future_return(df, n_bars)
    return (returns > threshold).astype(int)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/unknown965/coding/OpenCodeTest/crypto-invest-analysis && .venv/bin/python -m pytest tests/test_feature_engine_labels.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add feature_engine/labels.py tests/test_feature_engine_labels.py
git commit -m "feat: add feature_engine/labels.py for label generation"
```

---

## Task 4: Create feature_engine/builder.py — Assemble Feature Matrix

**Files:**
- Create: `feature_engine/builder.py`
- Create: `tests/test_feature_engine_builder.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_feature_engine_builder.py
import pandas as pd
import numpy as np
from feature_engine.builder import build_feature_matrix


def test_build_feature_matrix_returns_correct_shape():
    dates = pd.date_range("2024-01-01", periods=200, freq="1h")
    df = pd.DataFrame({
        "open": np.random.uniform(40000, 50000, 200),
        "high": np.random.uniform(40000, 50000, 200),
        "low": np.random.uniform(40000, 50000, 200),
        "close": np.random.uniform(40000, 50000, 200),
        "volume": np.random.uniform(100, 1000, 200),
    }, index=dates)
    
    features, labels = build_feature_matrix(df, n_bars=5)
    
    assert not features.empty
    assert not labels.empty
    assert len(features) == len(labels)


def test_build_feature_matrix_handles_nan():
    dates = pd.date_range("2024-01-01", periods=200, freq="1h")
    df = pd.DataFrame({
        "open": np.random.uniform(40000, 50000, 200),
        "high": np.random.uniform(40000, 50000, 200),
        "low": np.random.uniform(40000, 50000, 200),
        "close": np.random.uniform(40000, 50000, 200),
        "volume": np.random.uniform(100, 1000, 200),
    }, index=dates)
    
    features, labels = build_feature_matrix(df, n_bars=5)
    
    assert features.isna().sum().sum() == 0
    assert labels.isna().sum() == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/unknown965/coding/OpenCodeTest/crypto-invest-analysis && .venv/bin/python -m pytest tests/test_feature_engine_builder.py -v`
Expected: FAIL with "ImportError: cannot import name 'build_feature_matrix' from 'feature_engine.builder'"

- [ ] **Step 3: Write minimal implementation**

```python
# feature_engine/builder.py
import pandas as pd
from feature_engine.indicators import compute_all_indicators
from feature_engine.momentum import momentum_score, momentum_delta, momentum_acceleration
from feature_engine.labels import binary_label


def build_feature_matrix(df: pd.DataFrame, n_bars: int = 5) -> tuple:
    """Build complete feature matrix with labels.
    
    Args:
        df: OHLCV DataFrame
        n_bars: Number of bars to look ahead for labels
        
    Returns:
        Tuple of (features DataFrame, labels Series)
    """
    # Compute indicators
    indicators = compute_all_indicators(df)
    
    # Compute momentum
    momentum = momentum_score(indicators)
    delta = momentum_delta(momentum)
    acceleration = momentum_acceleration(delta)
    
    # Add momentum features
    indicators["momentum_score"] = momentum
    indicators["momentum_delta"] = delta
    indicators["momentum_acceleration"] = acceleration
    
    # Generate labels
    labels = binary_label(df, n_bars=n_bars)
    
    # Combine features
    features = indicators.copy()
    
    # Drop rows with NaN (first 200 periods for indicators, last n_bars for labels)
    valid_idx = features.dropna().index.intersection(labels.dropna().index)
    features = features.loc[valid_idx]
    labels = labels.loc[valid_idx]
    
    return features, labels
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/unknown965/coding/OpenCodeTest/crypto-invest-analysis && .venv/bin/python -m pytest tests/test_feature_engine_builder.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add feature_engine/builder.py tests/test_feature_engine_builder.py
git commit -m "feat: add feature_engine/builder.py for feature matrix assembly"
```

---

## Task 5: Create backtest_engine/engine.py — Core Backtest Logic

**Files:**
- Create: `backtest_engine/__init__.py`
- Create: `backtest_engine/engine.py`
- Create: `backtest_engine/strategy.py` (Strategy ABC)
- Create: `tests/test_backtest_engine.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_backtest_engine.py
import pandas as pd
import numpy as np
from backtest_engine.engine import BacktestEngine
from backtest_engine.strategy import Strategy, Signal


class DummyStrategy(Strategy):
    def evaluate(self, features: pd.Series) -> Signal:
        if features.get("momentum_score", 0) > 0.5:
            return Signal("偏多", 0.8, "dummy")
        elif features.get("momentum_score", 0) < -0.5:
            return Signal("偏空", 0.8, "dummy")
        return Signal("中立", 0.5, "dummy")


def test_backtest_engine_runs():
    dates = pd.date_range("2024-01-01", periods=200, freq="1h")
    features = pd.DataFrame({
        "momentum_score": np.sin(np.linspace(0, 10, 200)),
    }, index=dates)
    
    engine = BacktestEngine(
        strategy=DummyStrategy(),
        initial_capital=10000,
    )
    
    result = engine.run(features)
    
    assert result.total_trades >= 0
    assert result.final_equity > 0


def test_backtest_engine_tracks_positions():
    dates = pd.date_range("2024-01-01", periods=100, freq="1h")
    features = pd.DataFrame({
        "momentum_score": [0.8] * 50 + [-0.8] * 50,
    }, index=dates)
    
    engine = BacktestEngine(
        strategy=DummyStrategy(),
        initial_capital=10000,
    )
    
    result = engine.run(features)
    
    assert result.total_trades >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/unknown965/coding/OpenCodeTest/crypto-invest-analysis && .venv/bin/python -m pytest tests/test_backtest_engine.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'backtest_engine'"

- [ ] **Step 3: Create backtest_engine/__init__.py**

```python
# backtest_engine/__init__.py
```

- [ ] **Step 4: Create backtest_engine/strategy.py**

```python
# backtest_engine/strategy.py
from dataclasses import dataclass
import abc
import pandas as pd


@dataclass
class Signal:
    direction: str  # "偏多", "偏空", "中立"
    confidence: float
    source: str


class Strategy(abc.ABC):
    @abc.abstractmethod
    def evaluate(self, features: pd.Series) -> Signal:
        ...
```

- [ ] **Step 5: Write minimal implementation**

```python
# backtest_engine/engine.py
from dataclasses import dataclass, field
from typing import List
import pandas as pd
from backtest_engine.strategy import Strategy, Signal


@dataclass
class Position:
    symbol: str
    side: str
    quantity: float
    entry_price: float
    current_price: float = 0.0
    
    @property
    def unrealized_pnl(self) -> float:
        if self.side == "long":
            return (self.current_price - self.entry_price) * self.quantity
        return (self.entry_price - self.current_price) * self.quantity


@dataclass
class BacktestResult:
    total_trades: int
    final_equity: float
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    win_rate: float
    equity_curve: list
    trades: list


@dataclass
class BacktestEngine:
    strategy: Strategy
    initial_capital: float = 10000.0
    fee_rate: float = 0.001
    slippage: float = 0.0005
    max_position_pct: float = 25.0
    max_daily_trades: int = 10
    max_drawdown_stop: float = 30.0
    
    def run(self, features: pd.DataFrame) -> BacktestResult:
        cash = self.initial_capital
        positions: List[Position] = []
        equity_curve = []
        trades = []
        daily_trade_count = 0
        
        for i, (idx, row) in enumerate(features.iterrows()):
            sig = self.strategy.evaluate(row)
            price = row.get("close", 0)
            
            if price == 0:
                equity_curve.append(cash)
                continue
            
            # Check drawdown
            peak = max(equity_curve) if equity_curve else cash
            current_equity = cash + sum(p.quantity * price for p in positions)
            drawdown = ((peak - current_equity) / peak * 100) if peak > 0 else 0
            
            if drawdown > self.max_drawdown_stop and positions:
                for pos in positions:
                    cash += pos.quantity * price * (1 - self.fee_rate)
                    trades.append({"action": "sell", "price": price, "reason": "drawdown_stop"})
                positions.clear()
            
            # Execute trades
            if sig.direction == "偏多" and not positions and daily_trade_count < self.max_daily_trades:
                qty = (cash * self.max_position_pct / 100) / price
                if qty > 0:
                    cost = qty * price * (1 + self.fee_rate)
                    if cost <= cash:
                        cash -= cost
                        positions.append(Position("BTC/USDT", "long", qty, price))
                        trades.append({"action": "buy", "price": price, "quantity": qty})
                        daily_trade_count += 1
            
            elif sig.direction == "偏空" and positions and daily_trade_count < self.max_daily_trades:
                for pos in positions:
                    cash += pos.quantity * price * (1 - self.fee_rate)
                    trades.append({"action": "sell", "price": price, "quantity": pos.quantity})
                positions.clear()
                daily_trade_count += 1
            
            # Update equity
            current_equity = cash + sum(p.quantity * price for p in positions)
            equity_curve.append(current_equity)
        
        # Calculate metrics
        final_equity = equity_curve[-1] if equity_curve else cash
        total_return = ((final_equity - self.initial_capital) / self.initial_capital) * 100
        peak = max(equity_curve) if equity_curve else cash
        max_dd = ((peak - min(equity_curve)) / peak * 100) if equity_curve and peak > 0 else 0
        
        wins = sum(1 for t in trades if t["action"] == "sell" and t.get("pnl", 0) > 0)
        total_sells = sum(1 for t in trades if t["action"] == "sell")
        win_rate = (wins / total_sells * 100) if total_sells > 0 else 0
        
        # Calculate Sharpe ratio
        if len(equity_curve) > 1:
            returns = pd.Series(equity_curve).pct_change().dropna()
            if returns.std() > 0:
                sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * 24)  # Annualized for hourly
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
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd /Users/unknown965/coding/OpenCodeTest/crypto-invest-analysis && .venv/bin/python -m pytest tests/test_backtest_engine.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add backtest_engine/__init__.py backtest_engine/strategy.py backtest_engine/engine.py tests/test_backtest_engine.py
git commit -m "feat: add backtest_engine with core backtest logic and Strategy ABC"
```

---

## Task 6: Create backtest_engine/rule_strategy.py — Rule-Based Strategy

**Files:**
- Create: `backtest_engine/rule_strategy.py`
- Create: `tests/test_rule_strategy.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rule_strategy.py
import pandas as pd
import numpy as np
from backtest_engine.rule_strategy import MomentumRuleStrategy


def test_momentum_rule_buy_signal():
    features = pd.Series({
        "momentum_score": 0.7,
        "momentum_delta": 0.1,
    })
    
    strategy = MomentumRuleStrategy(buy_threshold=0.6, sell_threshold=-0.6)
    signal = strategy.evaluate(features)
    
    assert signal.direction == "偏多"


def test_momentum_rule_sell_signal():
    features = pd.Series({
        "momentum_score": -0.7,
        "momentum_delta": -0.1,
    })
    
    strategy = MomentumRuleStrategy(buy_threshold=0.6, sell_threshold=-0.6)
    signal = strategy.evaluate(features)
    
    assert signal.direction == "偏空"


def test_momentum_rule_neutral():
    features = pd.Series({
        "momentum_score": 0.0,
        "momentum_delta": 0.0,
    })
    
    strategy = MomentumRuleStrategy(buy_threshold=0.6, sell_threshold=-0.6)
    signal = strategy.evaluate(features)
    
    assert signal.direction == "中立"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/unknown965/coding/OpenCodeTest/crypto-invest-analysis && .venv/bin/python -m pytest tests/test_rule_strategy.py -v`
Expected: FAIL with "ImportError: cannot import name 'MomentumRuleStrategy' from 'backtest_engine.rule_strategy'"

- [ ] **Step 3: Write minimal implementation**

```python
# backtest_engine/rule_strategy.py
from backtest_engine.strategy import Strategy, Signal


class MomentumRuleStrategy(Strategy):
    def __init__(self, buy_threshold: float = 0.6, sell_threshold: float = -0.6):
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
    
    def evaluate(self, features) -> Signal:
        score = features.get("momentum_score", 0)
        delta = features.get("momentum_delta", 0)
        
        if score > self.buy_threshold and delta > 0:
            return Signal("偏多", abs(score), "momentum_rule")
        elif score < self.sell_threshold and delta < 0:
            return Signal("偏空", abs(score), "momentum_rule")
        else:
            return Signal("中立", 0.5, "momentum_rule")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/unknown965/coding/OpenCodeTest/crypto-invest-analysis && .venv/bin/python -m pytest tests/test_rule_strategy.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backtest_engine/rule_strategy.py tests/test_rule_strategy.py
git commit -m "feat: add backtest_engine/rule_strategy.py for rule-based trading"
```

---

## Task 7: Create backtest_engine/model_strategy.py — Model-Based Strategy

**Files:**
- Create: `backtest_engine/model_strategy.py`
- Create: `tests/test_model_strategy.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model_strategy.py
import pandas as pd
import numpy as np
from unittest.mock import Mock
from backtest_engine.model_strategy import ModelStrategy


def test_model_strategy_buy_signal():
    mock_model = Mock()
    mock_model.predict.return_value = np.array([1])
    
    features = pd.Series({"momentum_score": 0.5, "RSI": 60})
    
    strategy = ModelStrategy(model=mock_model, feature_columns=["momentum_score", "RSI"])
    signal = strategy.evaluate(features)
    
    assert signal.direction == "偏多"


def test_model_strategy_sell_signal():
    mock_model = Mock()
    mock_model.predict.return_value = np.array([0])
    
    features = pd.Series({"momentum_score": -0.5, "RSI": 40})
    
    strategy = ModelStrategy(model=mock_model, feature_columns=["momentum_score", "RSI"])
    signal = strategy.evaluate(features)
    
    assert signal.direction == "偏空"


def test_model_strategy_neutral_when_no_confidence():
    mock_model = Mock()
    mock_model.predict.return_value = np.array([1])
    mock_model.predict_proba.return_value = np.array([[0.51, 0.49]])
    
    features = pd.Series({"momentum_score": 0.0, "RSI": 50})
    
    strategy = ModelStrategy(
        model=mock_model,
        feature_columns=["momentum_score", "RSI"],
        confidence_threshold=0.6,
    )
    signal = strategy.evaluate(features)
    
    assert signal.direction == "中立"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/unknown965/coding/OpenCodeTest/crypto-invest-analysis && .venv/bin/python -m pytest tests/test_model_strategy.py -v`
Expected: FAIL with "ImportError: cannot import name 'ModelStrategy' from 'backtest_engine.model_strategy'"

- [ ] **Step 3: Write minimal implementation**

```python
# backtest_engine/model_strategy.py
import numpy as np
from backtest_engine.strategy import Strategy, Signal


class ModelStrategy(Strategy):
    def __init__(self, model, feature_columns: list, confidence_threshold: float = 0.5):
        self.model = model
        self.feature_columns = feature_columns
        self.confidence_threshold = confidence_threshold
    
    def evaluate(self, features) -> Signal:
        try:
            X = np.array([[features[col] for col in self.feature_columns]])
            prediction = self.model.predict(X)[0]
            
            # Try to get probability if available
            try:
                proba = self.model.predict_proba(X)[0]
                confidence = max(proba)
            except AttributeError:
                confidence = 1.0
            
            if confidence < self.confidence_threshold:
                return Signal("中立", confidence, "model")
            
            if prediction == 1:
                return Signal("偏多", confidence, "model")
            else:
                return Signal("偏空", confidence, "model")
                
        except Exception:
            return Signal("中立", 0.0, "model")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/unknown965/coding/OpenCodeTest/crypto-invest-analysis && .venv/bin/python -m pytest tests/test_model_strategy.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backtest_engine/model_strategy.py tests/test_model_strategy.py
git commit -m "feat: add backtest_engine/model_strategy.py for ML model trading"
```

---

## Task 8: Create backtest_engine/metrics.py — Performance Metrics

**Files:**
- Create: `backtest_engine/metrics.py`
- Create: `tests/test_backtest_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_backtest_metrics.py
import numpy as np
from backtest_engine.metrics import calculate_metrics


def test_calculate_metrics_basic():
    equity_curve = [10000, 10500, 11000, 10800, 11200]
    trades = [
        {"action": "buy", "price": 100},
        {"action": "sell", "price": 105, "pnl": 500},
        {"action": "buy", "price": 108},
        {"action": "sell", "price": 112, "pnl": 400},
    ]
    
    metrics = calculate_metrics(equity_curve, trades, initial_capital=10000)
    
    assert metrics["total_return_pct"] > 0
    assert metrics["max_drawdown_pct"] >= 0
    assert metrics["win_rate"] == 100.0


def test_calculate_metrics_with_losses():
    equity_curve = [10000, 9500, 9000, 9500, 10000]
    trades = [
        {"action": "buy", "price": 100},
        {"action": "sell", "price": 95, "pnl": -500},
        {"action": "buy", "price": 95},
        {"action": "sell", "price": 100, "pnl": 500},
    ]
    
    metrics = calculate_metrics(equity_curve, trades, initial_capital=10000)
    
    assert metrics["win_rate"] == 50.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/unknown965/coding/OpenCodeTest/crypto-invest-analysis && .venv/bin/python -m pytest tests/test_backtest_metrics.py -v`
Expected: FAIL with "ImportError: cannot import name 'calculate_metrics' from 'backtest_engine.metrics'"

- [ ] **Step 3: Write minimal implementation**

```python
# backtest_engine/metrics.py
import numpy as np


def calculate_metrics(equity_curve: list, trades: list, initial_capital: float) -> dict:
    """Calculate backtest performance metrics.
    
    Args:
        equity_curve: List of equity values over time
        trades: List of trade dictionaries
        initial_capital: Starting capital
        
    Returns:
        Dictionary of metrics
    """
    if not equity_curve:
        return {
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "win_rate": 0.0,
        }
    
    final_equity = equity_curve[-1]
    total_return = ((final_equity - initial_capital) / initial_capital) * 100
    
    # Calculate max drawdown
    peak = equity_curve[0]
    max_dd = 0.0
    for val in equity_curve:
        if val > peak:
            peak = val
        dd = (peak - val) / peak * 100
        if dd > max_dd:
            max_dd = dd
    
    # Calculate win rate
    sell_trades = [t for t in trades if t["action"] == "sell"]
    wins = sum(1 for t in sell_trades if t.get("pnl", 0) > 0)
    win_rate = (wins / len(sell_trades) * 100) if sell_trades else 0.0
    
    return {
        "total_return_pct": round(total_return, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "win_rate": round(win_rate, 1),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/unknown965/coding/OpenCodeTest/crypto-invest-analysis && .venv/bin/python -m pytest tests/test_backtest_metrics.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backtest_engine/metrics.py tests/test_backtest_metrics.py
git commit -m "feat: add backtest_engine/metrics.py for performance calculation"
```

---

## Task 9: Create Colab Notebook — ML Pipeline

**Files:**
- Create: `colab/momentum_ml.ipynb`

- [ ] **Step 1: Create notebook structure**

```python
# colab/momentum_ml.ipynb
# Cell 1: Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Cell 2: Load data
# Option A: Load from CSV
# df = pd.read_csv("BTC_USDT_1h_features.csv")
# Option B: Compute from scratch (upload feature_engine/ or clone repo)
from feature_engine.builder import build_feature_matrix
from data.fetcher import fetch_ohlcv

df_raw = fetch_ohlcv("BTC/USDT", timeframe="1h", limit=2000)
features, labels = build_feature_matrix(df_raw, n_bars=5)

# Cell 3: Split data
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, shuffle=False
)

# Cell 4: Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Cell 5: Evaluate
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))

# Cell 6: Feature importance
importance = pd.Series(rf.feature_importances_, index=features.columns)
importance.sort_values(ascending=False).head(10).plot(kind="barh")
plt.title("Top 10 Feature Importance")
plt.show()

# Cell 7: Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# Cell 8: Save model
import joblib
joblib.dump(rf, "momentum_rf_model.pkl")
print("Model saved to momentum_rf_model.pkl")
```

- [ ] **Step 2: Commit**

```bash
git add colab/momentum_ml.ipynb
git commit -m "feat: add Colab notebook for ML pipeline"
```

---

## Task 10: Run All Tests and Verify

- [ ] **Step 1: Run all new tests**

Run: `cd /Users/unknown965/coding/OpenCodeTest/crypto-invest-analysis && .venv/bin/python -m pytest tests/test_feature_engine_indicators.py tests/test_feature_engine_momentum.py tests/test_feature_engine_labels.py tests/test_feature_engine_builder.py tests/test_backtest_engine.py tests/test_rule_strategy.py tests/test_model_strategy.py tests/test_backtest_metrics.py -v`
Expected: All PASS

- [ ] **Step 2: Run existing tests to ensure no regression**

Run: `cd /Users/unknown965/coding/OpenCodeTest/crypto-invest-analysis && .venv/bin/python -m pytest tests/ -v`
Expected: All PASS (existing + new)

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "feat: complete local backtesting system implementation"
```

---

## Summary

| Task | Module | Description |
|------|--------|-------------|
| 1 | feature_engine/indicators.py | 18 technical indicators |
| 2 | feature_engine/momentum.py | Momentum quantification [-1, 1] |
| 3 | feature_engine/labels.py | Label generation |
| 4 | feature_engine/builder.py | Feature matrix assembly |
| 5 | backtest_engine/engine.py | Core backtest logic |
| 6 | backtest_engine/rule_strategy.py | Rule-based strategy |
| 7 | backtest_engine/model_strategy.py | Model-based strategy |
| 8 | backtest_engine/metrics.py | Performance metrics |
| 9 | colab/momentum_ml.ipynb | ML pipeline notebook |
| 10 | — | Integration testing |
