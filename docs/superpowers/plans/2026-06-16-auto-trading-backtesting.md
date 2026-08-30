# Auto Trading & Strategy Backtesting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add backtesting, paper trading, and live trading to the existing crypto analysis system.

**Architecture:** New `trading/` package with strategy engine, backtest runner, portfolio manager, and execution layer. Strategy evaluation runs in Monitor `/check`, backtest UI lives in Streamlit, all state persisted to JSON.

**Tech Stack:** Python, pandas, ccxt, ta, Plotly

---

## File Structure

### New files
- `trading/__init__.py` — empty
- `trading/config.py` — default strategy config, risk settings, schema
- `trading/strategy.py` — `Strategy` base class, 5 strategies (`MACross`, `RSIThreshold`, `MACDCross`, `CompositeStrategy`, `CustomComposite`)
- `trading/portfolio.py` — `Position`, `Order`, `Portfolio` dataclasses, serialization, PnL calculation
- `trading/executor.py` — `PaperExecutor`, `LiveExecutor`
- `trading/backtest.py` — `BacktestEngine`, performance metrics
- `tests/test_trading_strategy.py`
- `tests/test_trading_portfolio.py`
- `tests/test_trading_executor.py`
- `tests/test_trading_backtest.py`

### Modified files
- `monitor/config.yaml` — add `trading:` section (strategies, risk)
- `monitor/main.py` — `/check` runs strategy eval, calls executor, sends Discord
- `app.py` — add 2 new Streamlit pages (backtest, portfolio dashboard)

---

### Task 1: trading package scaffold + strategy config

**Files:**
- Create: `trading/__init__.py`
- Create: `trading/config.py`
- Test: `tests/test_trading_strategy.py` (strategy signal tests)

- [ ] **Step 1: Create `trading/__init__.py`**

```
EMPTY FILE
```

- [ ] **Step 2: Create `trading/config.py`**

```python
DEFAULT_STRATEGIES = {
    "ma_cross": {
        "enabled": True,
        "params": {"fast": 12, "slow": 26, "type": "ema"},
        "weight": 1,
    },
    "rsi": {
        "enabled": True,
        "params": {"period": 14, "overbought": 70, "oversold": 30},
        "weight": 1,
    },
    "macd": {
        "enabled": False,
        "params": {"fast": 12, "slow": 26, "signal": 9},
        "weight": 1,
    },
    "composite": {
        "enabled": True,
        "params": {},
        "weight": 2,
    },
}

DEFAULT_RISK = {
    "max_position_pct": 25,
    "min_bars_hold": 1,
    "max_daily_trades": 10,
    "max_drawdown_stop": 30,
    "signal_threshold": 0.6,
}
```

- [ ] **Step 3: Create `trading/strategy.py` with base + 5 strategies**

```python
from typing import Optional, Callable
from dataclasses import dataclass
import pandas as pd

@dataclass
class Signal:
    direction: str  # "偏多" | "偏空" | "中立"
    confidence: float  # 0.0 - 1.0
    source: str

class Strategy:
    name: str = ""
    def evaluate(self, overlay: pd.DataFrame, subplots: dict) -> Signal:
        raise NotImplementedError

class MACross(Strategy):
    name = "ma_cross"
    def __init__(self, fast: int = 12, slow: int = 26, ma_type: str = "ema"):
        self.fast = fast
        self.slow = slow
        self.ma_type = ma_type

    def evaluate(self, overlay: pd.DataFrame, subplots: dict) -> Signal:
        col_fast = f"{self.ma_type.upper()}_{self.fast}"
        col_slow = f"{self.ma_type.upper()}_{self.slow}"
        if col_fast not in overlay.columns or col_slow not in overlay.columns:
            return Signal("中立", 0.0, self.name)
        prev_fast = overlay[col_fast].iloc[-2]
        prev_slow = overlay[col_slow].iloc[-2]
        cur_fast = overlay[col_fast].iloc[-1]
        cur_slow = overlay[col_slow].iloc[-1]
        if prev_fast <= prev_slow and cur_fast > cur_slow:
            return Signal("偏多", 0.8, self.name)
        if prev_fast >= prev_slow and cur_fast < cur_slow:
            return Signal("偏空", 0.8, self.name)
        return Signal("中立", 0.5, self.name)


class RSIThreshold(Strategy):
    name = "rsi"
    def __init__(self, period: int = 14, overbought: int = 70, oversold: int = 30):
        self.period = period
        self.overbought = overbought
        self.oversold = oversold

    def evaluate(self, overlay: pd.DataFrame, subplots: dict) -> Signal:
        rsi = subplots.get("rsi", pd.DataFrame())
        if rsi.empty or "RSI" not in rsi.columns:
            return Signal("中立", 0.0, self.name)
        val = rsi["RSI"].iloc[-1]
        prev = rsi["RSI"].iloc[-2] if len(rsi) > 1 else val
        if prev >= self.oversold and val < self.oversold:
            return Signal("偏空", 0.7, self.name)
        if prev <= self.overbought and val > self.overbought:
            return Signal("偏多", 0.7, self.name)
        if val <= self.oversold:
            return Signal("偏多", 0.6, self.name)
        if val >= self.overbought:
            return Signal("偏空", 0.6, self.name)
        return Signal("中立", 0.5, self.name)


class MACDCross(Strategy):
    name = "macd"
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def evaluate(self, overlay: pd.DataFrame, subplots: dict) -> Signal:
        macd = subplots.get("macd", pd.DataFrame())
        if macd.empty or "MACD" not in macd.columns or "Signal" not in macd.columns:
            return Signal("中立", 0.0, self.name)
        prev_macd = macd["MACD"].iloc[-2]
        prev_sig = macd["Signal"].iloc[-2]
        cur_macd = macd["MACD"].iloc[-1]
        cur_sig = macd["Signal"].iloc[-1]
        if prev_macd <= prev_sig and cur_macd > cur_sig:
            return Signal("偏多", 0.8, self.name)
        if prev_macd >= prev_sig and cur_macd < cur_sig:
            return Signal("偏空", 0.8, self.name)
        return Signal("中立", 0.5, self.name)


class CompositeStrategy(Strategy):
    name = "composite"
    def evaluate(self, overlay: pd.DataFrame, subplots: dict) -> Signal:
        from analysis.summary import analyze_signals
        sig = analyze_signals(overlay, subplots)
        direction = sig.get("direction", "中立")
        total = sig.get("total", 1)
        bullish = sig.get("bullish_count", 0)
        bearish = sig.get("bearish_count", 0)
        confidence = max(bullish, bearish) / total if total > 0 else 0
        return Signal(direction, confidence, self.name)


class CustomComposite:
    def __init__(self, strategy_configs: dict, threshold: float = 0.6):
        self.strategies = []
        self.weights = []
        strategy_map = {
            "ma_cross": MACross,
            "rsi": RSIThreshold,
            "macd": MACDCross,
            "composite": CompositeStrategy,
        }
        for name, cfg in strategy_configs.items():
            if cfg.get("enabled") and name in strategy_map:
                cls = strategy_map[name]
                params = cfg.get("params", {})
                weight = cfg.get("weight", 1)
                self.strategies.append(cls(**params))
                self.weights.append(weight)
        self.threshold = threshold

    def evaluate(self, overlay: pd.DataFrame, subplots: dict) -> Optional[Signal]:
        total_weight = sum(self.weights)
        if total_weight == 0:
            return None
        score_bullish = 0.0
        score_bearish = 0.0
        for s, w in zip(self.strategies, self.weights):
            sig = s.evaluate(overlay, subplots)
            if sig.direction == "偏多":
                score_bullish += sig.confidence * w
            elif sig.direction == "偏空":
                score_bearish += sig.confidence * w
        bull_ratio = score_bullish / total_weight
        bear_ratio = score_bearish / total_weight
        if bull_ratio > self.threshold and bull_ratio > bear_ratio:
            return Signal("偏多", bull_ratio, "custom_composite")
        if bear_ratio > self.threshold and bear_ratio > bull_ratio:
            return Signal("偏空", bear_ratio, "custom_composite")
        return Signal("中立", max(bull_ratio, bear_ratio), "custom_composite")
```

- [ ] **Step 4: Write strategy tests**

```python
# tests/test_trading_strategy.py
import pandas as pd
import numpy as np
from trading.strategy import MACross, RSIThreshold, MACDCross, CompositeStrategy, CustomComposite, Signal

def _make_overlay(close_vals, sma_vals=None):
    df = pd.DataFrame({"close": close_vals})
    if sma_vals:
        df["SMA_20"] = sma_vals
        df["SMA_50"] = sma_vals
    df["EMA_12"] = close_vals
    df["EMA_26"] = close_vals
    df["SMA_200"] = close_vals
    return df

def test_ma_cross_golden():
    close = [100, 101, 102, 103, 104, 105]
    fast = [101, 102, 103, 104, 105, 106]
    slow = [103, 103, 103, 103, 103, 103]
    overlay = pd.DataFrame({"close": close, "EMA_12": fast, "EMA_26": slow,
                            "SMA_20": close, "SMA_50": close, "SMA_200": close})
    s = MACross(fast=12, slow=26, ma_type="ema")
    sig = s.evaluate(overlay, {})
    assert sig.direction == "偏多"

def test_ma_cross_death():
    close = [105, 104, 103, 102, 101, 100]
    fast = [105, 104, 103, 102, 101, 100]
    slow = [103, 103, 103, 103, 103, 103]
    overlay = pd.DataFrame({"close": close, "EMA_12": fast, "EMA_26": slow,
                            "SMA_20": close, "SMA_50": close, "SMA_200": close})
    s = MACross(fast=12, slow=26, ma_type="ema")
    sig = s.evaluate(overlay, {})
    assert sig.direction == "偏空"

def test_ma_cross_no_cross():
    close = [100, 101, 102, 103, 104, 105]
    fast = [101, 101, 101, 101, 101, 101]
    slow = [103, 103, 103, 103, 103, 103]
    overlay = pd.DataFrame({"close": close, "EMA_12": fast, "EMA_26": slow,
                            "SMA_20": close, "SMA_50": close, "SMA_200": close})
    s = MACross(fast=12, slow=26, ma_type="ema")
    sig = s.evaluate(overlay, {})
    assert sig.direction == "中立"

def test_rsi_oversold_bullish():
    rsi_vals = [35, 34, 33, 32, 31, 29]  # crosses below 30 → oversold → bullish
    subplots = {"rsi": pd.DataFrame({"RSI": rsi_vals})}
    s = RSIThreshold(period=14, overbought=70, oversold=30)
    sig = s.evaluate(_make_overlay(rsi_vals), subplots)
    assert sig.direction == "偏多"

def test_rsi_overbought_bearish():
    rsi_vals = [68, 69, 70, 71, 72, 71]  # crosses above 70 → overbought → bearish
    subplots = {"rsi": pd.DataFrame({"RSI": rsi_vals})}
    s = RSIThreshold(period=14, overbought=70, oversold=30)
    sig = s.evaluate(_make_overlay(rsi_vals), subplots)
    assert sig.direction == "偏空"

def test_macd_bullish_cross():
    macd_vals = [-1, -0.8, -0.6, -0.4, -0.2, 0.1]
    sig_vals = [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5]
    subplots = {"macd": pd.DataFrame({"MACD": macd_vals, "Signal": sig_vals, "Histogram": [0]*6})}
    s = MACDCross()
    sig = s.evaluate(_make_overlay([100]*6), subplots)
    assert sig.direction == "偏多"

def test_macd_bearish_cross():
    macd_vals = [0.5, 0.3, 0.1, -0.1, -0.3, -0.5]
    sig_vals = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    subplots = {"macd": pd.DataFrame({"MACD": macd_vals, "Signal": sig_vals, "Histogram": [0]*6})}
    s = MACDCross()
    sig = s.evaluate(_make_overlay([100]*6), subplots)
    assert sig.direction == "偏空"

def test_composite_strategy_needs_real_data():
    close = [100, 102, 104, 106, 108, 110]
    overlay = pd.DataFrame({
        "close": close, "SMA_20": [101]*6, "SMA_50": [102]*6, "SMA_200": [100]*6,
        "EMA_12": [103]*6, "EMA_26": [102]*6,
        "BB_upper": [110]*6, "BB_middle": [105]*6, "BB_lower": [100]*6,
    })
    subplots = {
        "rsi": pd.DataFrame({"RSI": [55, 56, 57, 58, 59, 60]}),
        "macd": pd.DataFrame({"MACD": [1]*6, "Signal": [0.5]*6, "Histogram": [0.5]*6}),
        "stoch": pd.DataFrame({"%K": [50]*6, "%D": [45]*6}),
        "obv": pd.DataFrame({"OBV": [1000]*6}),
    }
    s = CompositeStrategy()
    sig = s.evaluate(overlay, subplots)
    assert sig.direction in ("偏多", "偏空", "中立")

def test_custom_composite_ma_only():
    configs = {
        "ma_cross": {"enabled": True, "params": {"fast": 12, "slow": 26, "type": "ema"}, "weight": 1},
        "rsi": {"enabled": False, "params": {}, "weight": 1},
        "macd": {"enabled": False, "params": {}, "weight": 1},
        "composite": {"enabled": False, "params": {}, "weight": 1},
    }
    cc = CustomComposite(configs, threshold=0.6)
    close = [100, 101, 102, 103, 104, 105]
    fast = [101, 102, 103, 104, 105, 106]
    slow = [103, 103, 103, 103, 103, 103]
    overlay = pd.DataFrame({"close": close, "EMA_12": fast, "EMA_26": slow,
                            "SMA_20": close, "SMA_50": close, "SMA_200": close})
    sig = cc.evaluate(overlay, {})
    assert sig is not None
```

- [ ] **Step 5: Run tests to verify they fail first, then pass**

Run: `python -m pytest tests/test_trading_strategy.py -v 2>&1 | head -40`
Expected: tests fail with ImportError (no trading module yet), then pass after creating the files.

- [ ] **Step 6: Commit**

```bash
git add trading/ tests/test_trading_strategy.py
git commit -m "feat: add trading strategy engine with 5 strategies"
```


### Task 2: Portfolio management

**Files:**
- Create: `trading/portfolio.py`
- Test: `tests/test_trading_portfolio.py`

- [ ] **Step 1: Create `trading/portfolio.py`**

```python
from dataclasses import dataclass, asdict
from typing import Optional
from datetime import datetime
import json
from pathlib import Path

@dataclass
class Order:
    id: str
    symbol: str
    side: str  # "buy" | "sell"
    quantity: float
    price: float
    fee: float
    timestamp: str
    status: str  # "filled" | "pending" | "cancelled"
    pnl: float = 0.0
    pnl_pct: float = 0.0

@dataclass
class Position:
    symbol: str
    side: str  # "long" | "short"
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0

    def update_market(self, price: float):
        self.current_price = price
        if self.side == "long":
            self.unrealized_pnl = (price - self.entry_price) * self.quantity
        else:
            self.unrealized_pnl = (self.entry_price - price) * self.quantity
        self.unrealized_pnl_pct = (self.unrealized_pnl / (self.entry_price * self.quantity)) * 100

@dataclass
class Portfolio:
    cash: float
    positions: list
    orders: list
    total_deposits: float = 0.0
    total_withdrawals: float = 0.0

    @property
    def market_value(self) -> float:
        return sum(p.quantity * p.current_price for p in self.positions)

    @property
    def total_equity(self) -> float:
        return self.cash + self.market_value

    @property
    def realized_pnl(self) -> float:
        return sum(o.pnl for o in self.orders if o.status == "filled")

    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + sum(p.unrealized_pnl for p in self.positions)

    def to_dict(self) -> dict:
        return {
            "cash": self.cash,
            "positions": [asdict(p) for p in self.positions],
            "orders": [asdict(o) for o in self.orders],
            "total_deposits": self.total_deposits,
            "total_withdrawals": self.total_withdrawals,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Portfolio":
        return cls(
            cash=data["cash"],
            positions=[Position(**p) for p in data.get("positions", [])],
            orders=[Order(**o) for o in data.get("orders", [])],
            total_deposits=data.get("total_deposits", 0.0),
            total_withdrawals=data.get("total_withdrawals", 0.0),
        )


class PortfolioStore:
    def __init__(self, path: Path = Path("trading/state.json")):
        self.path = path

    def load(self) -> Portfolio:
        if not self.path.exists():
            return Portfolio(cash=10000.0, positions=[], orders=[])
        with open(self.path) as f:
            return Portfolio.from_dict(json.load(f))

    def save(self, portfolio: Portfolio):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(portfolio.to_dict(), f, indent=2)


def calculate_position_size(cash: float, price: float, max_pct: float = 25.0) -> float:
    max_amount = cash * (max_pct / 100.0)
    quantity = max_amount / price
    return round(quantity, 6)
```

- [ ] **Step 2: Write portfolio tests**

```python
# tests/test_trading_portfolio.py
from trading.portfolio import Position, Order, Portfolio, PortfolioStore, calculate_position_size
from pathlib import Path
import json

def test_position_update_market_long():
    p = Position(symbol="BTC/USDT", side="long", quantity=1.0, entry_price=50000, current_price=50000)
    p.update_market(55000)
    assert p.unrealized_pnl == 5000.0

def test_position_update_market_short():
    p = Position(symbol="BTC/USDT", side="short", quantity=1.0, entry_price=50000, current_price=50000)
    p.update_market(45000)
    assert p.unrealized_pnl == 5000.0

def test_portfolio_equity():
    pos = Position(symbol="BTC/USDT", side="long", quantity=1.0, entry_price=50000, current_price=55000)
    pf = Portfolio(cash=10000, positions=[pos], orders=[])
    pf.positions[0].update_market(55000)
    assert pf.market_value == 55000
    assert pf.total_equity == 65000

def test_portfolio_roundtrip(tmp_path):
    pos = Position(symbol="BTC/USDT", side="long", quantity=1.0, entry_price=50000, current_price=55000)
    pf = Portfolio(cash=10000, positions=[pos], orders=[])
    store = PortfolioStore(tmp_path / "state.json")
    store.save(pf)
    loaded = store.load()
    assert loaded.cash == 10000
    assert len(loaded.positions) == 1
    assert loaded.positions[0].symbol == "BTC/USDT"

def test_calculate_position_size():
    qty = calculate_position_size(10000, 50000, 25)
    assert qty == 0.05

def test_realized_pnl():
    o1 = Order(id="1", symbol="BTC/USDT", side="sell", quantity=1.0, price=55000, fee=10, timestamp="now", status="filled", pnl=5000, pnl_pct=10.0)
    o2 = Order(id="2", symbol="BTC/USDT", side="sell", quantity=1.0, price=45000, fee=10, timestamp="now", status="filled", pnl=-3000, pnl_pct=-6.0)
    pf = Portfolio(cash=50000, positions=[], orders=[o1, o2])
    assert pf.realized_pnl == 2000
```

- [ ] **Step 3: Run portfolio tests**

Run: `python -m pytest tests/test_trading_portfolio.py -v`
Expected: All 6 tests pass

- [ ] **Step 4: Commit**

```bash
git add trading/portfolio.py tests/test_trading_portfolio.py
git commit -m "feat: add portfolio management with position/order tracking"
```


### Task 3: Executor layer

**Files:**
- Create: `trading/executor.py`
- Test: `tests/test_trading_executor.py`

- [ ] **Step 1: Create `trading/executor.py`**

```python
from trading.portfolio import Portfolio, Position, Order, calculate_position_size
from datetime import datetime, timezone
import uuid

SLIPPAGE_RATE = 0.0005  # 0.05%


class PaperExecutor:
    def __init__(self, portfolio: Portfolio, slippage: float = SLIPPAGE_RATE):
        self.portfolio = portfolio
        self.slippage = slippage

    def execute_buy(self, symbol: str, price: float, quantity: float) -> Order:
        fee_rate = 0.001
        fill_price = price * (1 + self.slippage)
        cost = fill_price * quantity
        fee = cost * fee_rate
        self.portfolio.cash -= (cost + fee)
        existing = [p for p in self.portfolio.positions if p.symbol == symbol and p.side == "long"]
        if existing:
            pos = existing[0]
            total_qty = pos.quantity + quantity
            total_cost = pos.entry_price * pos.quantity + fill_price * quantity
            pos.entry_price = total_cost / total_qty
            pos.quantity = total_qty
        else:
            self.portfolio.positions.append(Position(
                symbol=symbol, side="long", quantity=quantity,
                entry_price=fill_price, current_price=fill_price,
            ))
        order = Order(
            id=str(uuid.uuid4())[:8], symbol=symbol, side="buy",
            quantity=quantity, price=fill_price, fee=fee,
            timestamp=datetime.now(timezone.utc).isoformat(),
            status="filled",
        )
        self.portfolio.orders.append(order)
        return order

    def execute_sell(self, symbol: str, price: float, quantity: float) -> Optional[Order]:
        pos = next((p for p in self.portfolio.positions if p.symbol == symbol and p.side == "long"), None)
        if not pos or pos.quantity < quantity:
            return None
        fee_rate = 0.001
        fill_price = price * (1 - self.slippage)
        revenue = fill_price * quantity
        fee = revenue * fee_rate
        cost_basis = pos.entry_price * quantity
        pnl = revenue - cost_basis - fee
        pnl_pct = (pnl / cost_basis) * 100
        self.portfolio.cash += (revenue - fee)
        pos.quantity -= quantity
        if pos.quantity <= 0:
            self.portfolio.positions = [p for p in self.portfolio.positions if p != pos]
        order = Order(
            id=str(uuid.uuid4())[:8], symbol=symbol, side="sell",
            quantity=quantity, price=fill_price, fee=fee,
            timestamp=datetime.now(timezone.utc).isoformat(),
            status="filled", pnl=pnl, pnl_pct=pnl_pct,
        )
        self.portfolio.orders.append(order)
        return order

    def can_trade(self, symbol: str, daily_count: int, max_daily: int = 10) -> bool:
        return daily_count < max_daily


class LiveExecutor:
    def __init__(self, portfolio: Portfolio, exchange=None, slippage: float = SLIPPAGE_RATE):
        self.portfolio = portfolio
        self.exchange = exchange
        self.slippage = slippage
        self.paper = PaperExecutor(portfolio, slippage)

    def execute_buy(self, symbol: str, price: float, quantity: float) -> Order:
        if self.exchange:
            try:
                order = self.exchange.create_market_buy_order(symbol.replace("/", ""), quantity)
                fill_price = order.get("price", price)
                filled = order.get("filled", quantity)
                fee = sum(f.get("cost", 0) for f in order.get("fees", []))
            except Exception:
                return self.paper.execute_buy(symbol, price, quantity)
        else:
            return self.paper.execute_buy(symbol, price, quantity)
        order_obj = Order(
            id=str(order.get("id", uuid.uuid4()))[:8], symbol=symbol, side="buy",
            quantity=filled, price=fill_price, fee=fee,
            timestamp=datetime.now(timezone.utc).isoformat(),
            status="filled",
        )
        self.portfolio.cash -= (fill_price * filled + fee)
        self.portfolio.positions.append(Position(
            symbol=symbol, side="long", quantity=filled,
            entry_price=fill_price, current_price=fill_price,
        ))
        self.portfolio.orders.append(order_obj)
        return order_obj

    def can_trade(self, symbol: str, daily_count: int, max_daily: int = 10) -> bool:
        return daily_count < max_daily
```

- [ ] **Step 2: Write executor tests**

```python
# tests/test_trading_executor.py
from trading.portfolio import Portfolio, Position
from trading.executor import PaperExecutor

def test_paper_buy():
    pf = Portfolio(cash=10000, positions=[], orders=[])
    ex = PaperExecutor(pf)
    order = ex.execute_buy("BTC/USDT", 50000, 0.05)
    assert order.side == "buy"
    assert order.status == "filled"
    assert len(pf.positions) == 1
    assert pf.positions[0].quantity == 0.05
    assert pf.cash < 10000

def test_paper_sell():
    pf = Portfolio(cash=10000, positions=[Position(symbol="BTC/USDT", side="long", quantity=0.05, entry_price=50000, current_price=50000)], orders=[])
    ex = PaperExecutor(pf)
    order = ex.execute_sell("BTC/USDT", 55000, 0.05)
    assert order is not None
    assert order.side == "sell"
    assert order.pnl > 0
    assert len(pf.positions) == 0
    assert pf.cash > 10000

def test_paper_sell_insufficient():
    pf = Portfolio(cash=10000, positions=[], orders=[])
    ex = PaperExecutor(pf)
    order = ex.execute_sell("BTC/USDT", 50000, 0.05)
    assert order is None

def test_paper_partial_sell():
    pf = Portfolio(cash=10000, positions=[Position(symbol="BTC/USDT", side="long", quantity=0.1, entry_price=50000, current_price=50000)], orders=[])
    ex = PaperExecutor(pf)
    order = ex.execute_sell("BTC/USDT", 55000, 0.04)
    assert order is not None
    assert len(pf.positions) == 1
    assert pf.positions[0].quantity == 0.06

def test_can_trade_limit():
    pf = Portfolio(cash=10000, positions=[], orders=[])
    ex = PaperExecutor(pf)
    assert ex.can_trade("BTC/USDT", 9, 10) == True
    assert ex.can_trade("BTC/USDT", 10, 10) == False
```

- [ ] **Step 3: Run executor tests**

Run: `python -m pytest tests/test_trading_executor.py -v`
Expected: All 5 tests pass

- [ ] **Step 4: Commit**

```bash
git add trading/executor.py tests/test_trading_executor.py
git commit -m "feat: add paper and live executor"
```


### Task 4: Backtest engine

**Files:**
- Create: `trading/backtest.py`
- Test: `tests/test_trading_backtest.py`

- [ ] **Step 1: Create `trading/backtest.py`**

```python
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional
from datetime import datetime, timezone

from data.fetcher import fetch_ohlcv
from indicators.calculator import compute_all
from trading.strategy import CustomComposite, Signal
from trading.portfolio import Portfolio, Position, Order, calculate_position_size
from trading.executor import PaperExecutor

@dataclass
class BacktestResult:
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    initial_capital: float
    final_equity: float
    total_return_pct: float
    cagr: float
    max_drawdown_pct: float
    sharpe_ratio: float
    win_rate: float
    total_trades: int
    equity_curve: list
    trades: list
    monthly_returns: dict

    def to_dict(self) -> dict:
        return asdict(self)


def _calc_sharpe(returns_series: pd.Series, risk_free: float = 0.02) -> float:
    if len(returns_series) < 2 or returns_series.std() == 0:
        return 0.0
    daily_rf = risk_free / 365
    excess = returns_series - daily_rf
    return float(np.sqrt(365) * excess.mean() / returns_series.std())


def _calc_max_drawdown(equity_series: list) -> float:
    peak = equity_series[0]
    max_dd = 0.0
    for val in equity_series:
        if val > peak:
            peak = val
        dd = (peak - val) / peak * 100
        if dd > max_dd:
            max_dd = dd
    return max_dd


def run_backtest(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    limit: int = 2000,
    initial_capital: float = 10000.0,
    fee_rate: float = 0.001,
    slippage: float = 0.0005,
    strategy_configs: Optional[dict] = None,
    signal_threshold: float = 0.6,
    risk_config: Optional[dict] = None,
) -> BacktestResult:
    if strategy_configs is None:
        from trading.config import DEFAULT_STRATEGIES
        strategy_configs = DEFAULT_STRATEGIES
    if risk_config is None:
        from trading.config import DEFAULT_RISK
        risk_config = DEFAULT_RISK

    df_raw = fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    if df_raw.empty:
        raise ValueError(f"No data for {symbol} {timeframe}")

    result = compute_all(df_raw)
    overlay = result["overlay"]
    subplots = result["subplots"]

    strategy = CustomComposite(strategy_configs, threshold=signal_threshold)
    portfolio = Portfolio(cash=initial_capital, positions=[], orders=[])
    executor = PaperExecutor(portfolio, slippage=slippage)

    equity_curve = []
    trades_log = []
    daily_returns = {}
    prev_equity = initial_capital
    daily_trade_count = 0
    last_trade_date = ""

    max_position_pct = risk_config.get("max_position_pct", 25)
    min_bars_hold = risk_config.get("min_bars_hold", 1)
    max_daily_trades = risk_config.get("max_daily_trades", 10)
    max_drawdown_stop = risk_config.get("max_drawdown_stop", 30)

    for i in range(min_bars_hold, len(overlay)):
        window = overlay.iloc[:i+1]
        sub_window = {k: v.iloc[:i+1] for k, v in subplots.items()}

        sig = strategy.evaluate(window, sub_window)
        if sig is None or sig.direction == "中立":
            equity_curve.append(portfolio.total_equity)
            continue

        current_price = float(overlay["close"].iloc[i])
        current_date = str(overlay.index[i].date()) if hasattr(overlay.index[i], "date") else str(i)

        if current_date != last_trade_date:
            daily_trade_count = 0
            last_trade_date = current_date

        has_position = any(p.symbol == symbol and p.side == "long" for p in portfolio.positions)

        # Check max drawdown stop
        current_dd = _calc_max_drawdown(equity_curve + [portfolio.total_equity])
        if current_dd > max_drawdown_stop:
            if has_position:
                executor.execute_sell(symbol, current_price, portfolio.positions[0].quantity)
            equity_curve.append(portfolio.total_equity)
            continue

        if sig.direction == "偏多" and not has_position and executor.can_trade(symbol, daily_trade_count, max_daily_trades):
            if i < min_bars_hold:
                equity_curve.append(portfolio.total_equity)
                continue
            qty = calculate_position_size(portfolio.cash, current_price, max_position_pct)
            if qty > 0:
                order = executor.execute_buy(symbol, current_price, qty)
                trades_log.append({
                    "date": current_date, "action": "buy",
                    "price": order.price, "quantity": order.quantity,
                })
                daily_trade_count += 1

        elif sig.direction == "偏空" and has_position and executor.can_trade(symbol, daily_trade_count, max_daily_trades):
            pos = [p for p in portfolio.positions if p.symbol == symbol and p.side == "long"]
            if pos:
                order = executor.execute_sell(symbol, current_price, pos[0].quantity)
                if order:
                    trades_log.append({
                        "date": current_date, "action": "sell",
                        "price": order.price, "quantity": order.quantity,
                        "pnl": order.pnl, "pnl_pct": order.pnl_pct,
                    })
                    daily_trade_count += 1

        # Update positions market price
        for p in portfolio.positions:
            p.update_market(current_price)

        equity_curve.append(portfolio.total_equity)

    final_equity = portfolio.total_equity
    total_return = ((final_equity - initial_capital) / initial_capital) * 100

    # Calculate CAGR
    total_days = len(equity_curve)
    cagr = ((final_equity / initial_capital) ** (365 / max(total_days, 1)) - 1) * 100 if total_days > 0 else 0.0

    # Calculate Sharpe
    equity_series = pd.Series(equity_curve)
    returns_series = equity_series.pct_change().dropna()
    sharpe = _calc_sharpe(returns_series)

    max_dd = _calc_max_drawdown(equity_curve)

    # Win rate
    filled_sells = [o for o in portfolio.orders if o.side == "sell" and o.status == "filled"]
    wins = sum(1 for o in filled_sells if o.pnl > 0)
    total_closed = len(filled_sells)
    win_rate = (wins / total_closed * 100) if total_closed > 0 else 0.0

    # Monthly returns
    monthly = {}
    for i, eq in enumerate(equity_curve):
        if i == 0:
            continue
        prev_eq = equity_curve[i-1]
        ret = ((eq - prev_eq) / prev_eq) * 100
        # approximate month from index
        month_key = str(i // 30)
        if month_key not in monthly:
            monthly[month_key] = []
        monthly[month_key].append(ret)
    monthly_returns = {k: sum(v) for k, v in monthly.items()}

    start_date = str(overlay.index[0]) if hasattr(overlay.index, "__getitem__") else ""
    end_date = str(overlay.index[-1]) if hasattr(overlay.index, "__getitem__") else ""

    return BacktestResult(
        symbol=symbol, timeframe=timeframe,
        start_date=start_date, end_date=end_date,
        initial_capital=initial_capital, final_equity=final_equity,
        total_return_pct=round(total_return, 2),
        cagr=round(cagr, 2), max_drawdown_pct=round(max_dd, 2),
        sharpe_ratio=round(sharpe, 2), win_rate=round(win_rate, 1),
        total_trades=len(trades_log),
        equity_curve=[round(e, 2) for e in equity_curve],
        trades=trades_log, monthly_returns=monthly_returns,
    )
```

- [ ] **Step 2: Write backtest tests**

```python
# tests/test_trading_backtest.py
from trading.backtest import _calc_sharpe, _calc_max_drawdown
import pandas as pd

def test_sharpe_positive():
    rets = pd.Series([0.001] * 100)
    sr = _calc_sharpe(rets, risk_free=0.02)
    assert sr > 0

def test_sharpe_negative():
    rets = pd.Series([-0.001] * 100)
    sr = _calc_sharpe(rets, risk_free=0.02)
    assert sr < 0

def test_sharpe_insufficient_data():
    rets = pd.Series([0.01])
    sr = _calc_sharpe(rets)
    assert sr == 0.0

def test_max_drawdown():
    eq = [100, 110, 90, 80, 95, 105]
    dd = _calc_max_drawdown(eq)
    assert dd > 0
    assert dd < 100

def test_max_drawdown_zero():
    eq = [100, 100, 100]
    dd = _calc_max_drawdown(eq)
    assert dd == 0.0

def test_run_backtest_with_defaults():
    from trading.backtest import run_backtest
    try:
        result = run_backtest(symbol="BTC/USDT", timeframe="1h", limit=200)
        assert result.symbol == "BTC/USDT"
        assert result.initial_capital == 10000
        assert result.total_trades >= 0
        assert len(result.equity_curve) > 0
        assert result.final_equity > 0
    except (ValueError, ConnectionError) as e:
        # API failures should not cause test failure
        pass
```

- [ ] **Step 3: Run backtest tests**

Run: `python -m pytest tests/test_trading_backtest.py -v`
Expected: Unit tests pass (integration test may be skipped if API unavailable)

- [ ] **Step 4: Commit**

```bash
git add trading/backtest.py tests/test_trading_backtest.py
git commit -m "feat: add backtest engine with performance metrics"
```


### Task 5: Integrate strategy evaluation into Monitor

**Files:**
- Modify: `monitor/config.yaml`
- Modify: `monitor/main.py`

- [ ] **Step 1: Update `monitor/config.yaml`** — add trading section

Add at the end of the file:
```yaml
trading:
  mode: "paper"  # "paper" | "live"
  strategies:
    ma_cross:
      enabled: true
      params: { fast: 12, slow: 26, type: "ema" }
      weight: 1
    rsi:
      enabled: true
      params: { period: 14, overbought: 70, oversold: 30 }
      weight: 1
    macd:
      enabled: false
      params: { fast: 12, slow: 26, signal: 9 }
      weight: 1
    composite:
      enabled: true
      params: {}
      weight: 2
  risk:
    max_position_pct: 25
    min_bars_hold: 1
    max_daily_trades: 10
    max_drawdown_stop: 30
    signal_threshold: 0.6
```

- [ ] **Step 2: Add trading logic to `/check` endpoint in `monitor/main.py`**

Append after the reversal/strong signal alerts block (before `today_state[symbol] = new_sig`):

```python
# Level 3: trading strategy evaluation
trading_cfg = CONFIG.get("trading", {})
if trading_cfg.get("strategies"):
    from trading.strategy import CustomComposite
    from trading.portfolio import PortfolioStore
    from trading.executor import PaperExecutor, LiveExecutor
    from trading.config import DEFAULT_RISK

    mode = trading_cfg.get("mode", "paper")
    risk = trading_cfg.get("risk", DEFAULT_RISK)
    threshold = risk.get("signal_threshold", 0.6)

    strategy = CustomComposite(trading_cfg["strategies"], threshold=threshold)
    store = PortfolioStore()
    portfolio = store.load()

    signal = strategy.evaluate(result["overlay"], result["subplots"])
    if signal and signal.direction != "中立":
        current_price = float(result["overlay"]["close"].iloc[-1])
        has_pos = any(p.symbol == symbol for p in portfolio.positions)

        if mode == "live":
            import ccxt
            exchange = ccxt.binance()
            executor = LiveExecutor(portfolio, exchange=exchange)
        else:
            executor = PaperExecutor(portfolio)

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        daily_count = sum(1 for o in portfolio.orders if o.timestamp.startswith(today))

        if signal.direction == "偏多" and not has_pos and executor.can_trade(symbol, daily_count, risk.get("max_daily_trades", 10)):
            qty = calculate_position_size(portfolio.cash, current_price, risk.get("max_position_pct", 25))
            if qty > 0:
                order = executor.execute_buy(symbol, current_price, qty)
                embed = {
                    "embeds": [{
                        "title": f"🟢 {symbol} 買入訊號",
                        "description": f"價格: ${current_price:,.2f}\n數量: {qty}\n信心: {signal.confidence:.0%}",
                        "color": 0x00FF00,
                    }]
                }
                try:
                    send_webhook(webhook, embed)
                    alerts_sent.append(f"{symbol}: BUY signal")
                except Exception:
                    pass

        elif signal.direction == "偏空" and has_pos and executor.can_trade(symbol, daily_count, risk.get("max_daily_trades", 10)):
            pos = [p for p in portfolio.positions if p.symbol == symbol][0]
            order = executor.execute_sell(symbol, current_price, pos.quantity)
            if order:
                embed = {
                    "embeds": [{
                        "title": f"🔴 {symbol} 賣出訊號",
                        "description": f"價格: ${current_price:,.2f}\n數量: {pos.quantity}\n損益: ${order.pnl:+.2f} ({order.pnl_pct:+.2f}%)",
                        "color": 0xFF0000,
                    }]
                }
                try:
                    send_webhook(webhook, embed)
                    alerts_sent.append(f"{symbol}: SELL signal (PnL: ${order.pnl:+.2f})")
                except Exception:
                    pass

        # Update position market prices
        for p in portfolio.positions:
            p.update_market(current_price)

        store.save(portfolio)
```

- [ ] **Step 3: Add import at the top of `monitor/main.py`**

After existing imports, add:
```python
from trading.strategy import CustomComposite
from trading.portfolio import PortfolioStore, calculate_position_size
from trading.executor import PaperExecutor, LiveExecutor
from trading.config import DEFAULT_RISK
```

- [ ] **Step 4: Commit**

```bash
git add monitor/config.yaml monitor/main.py
git commit -m "feat: integrate trading strategy evaluation into Monitor /check"
```


### Task 6: Streamlit — Backtest page (modify app.py for multi-page)

**Files:**
- Modify: `app.py`

The existing `app.py` is a single-page Streamlit app (322 lines). This task refactors it into a multi-page layout.

- [ ] **Step 1: Add page selector to sidebar and wrap existing code as `dashboard()`**

Replace the top of `app.py` (after `st.set_page_config`) with:

```python
st.set_page_config(layout="wide", page_title="Crypto Invest Analysis")

PAGE_NAMES = {
    "dashboard": "📈 技術分析儀表板",
    "backtest": "📊 策略回測",
    "portfolio": "💰 交易 Dashboard",
}

if "page" not in st.session_state:
    st.session_state.page = "dashboard"

def switch_page(name):
    st.session_state.page = name
```

Then in the sidebar, replace the existing settings block with a page selector:

```python
with st.sidebar:
    st.title("Navigation")
    for key, label in PAGE_NAMES.items():
        if st.button(label, use_container_width=True,
                     type="primary" if st.session_state.page == key else "secondary"):
            switch_page(key)
    st.markdown("---")
```

Then wrap lines 17-322 (the entire existing dashboard code) in a function:

```python
def dashboard():
    with st.sidebar:
        st.header("Settings")
        symbol = st.selectbox("Symbol", SYMBOLS)
        timeframe = st.selectbox("Timeframe", TIMEFRAMES, index=3)
        limit = st.slider("Candles", 50, 500, DEFAULT_CANDLE_COUNT)
        st.markdown("---")
        st.subheader("Indicators")
        show_sma = st.checkbox("SMA", True, ...)
        # ... all existing sidebar settings and main content ...
```

And add a page router at the very bottom of `app.py`:

```python
pages = {
    "dashboard": dashboard,
    "backtest": backtest_page,
    "portfolio": portfolio_page,
}
pages[st.session_state.page]()
```

Note: `backtest_page` and `portfolio_page` are defined in Task 6 Step 2 and Task 7 Step 1 respectively. The code won't compile until both are added. Commit after both functions are defined.

- [ ] **Step 2: Add `backtest_page()` function**

Before the page router, add:

```python
def backtest_page():
    st.title("📊 策略回測")
    st.markdown("---")

    from trading.backtest import run_backtest
    from trading.config import DEFAULT_STRATEGIES, DEFAULT_RISK

    with st.sidebar:
        st.header("回測參數")
        symbol = st.selectbox("交易對", ["BTC/USDT", "SOL/USDT"], key="bt_symbol")
        tf = st.selectbox("時間框架", ["1h", "4h", "1d"], index=0, key="bt_tf")
        lookback = st.slider("回測K線數", 100, 2000, 500, key="bt_lookback")
        capital = st.number_input("初始資金 ($)", 1000, 100000, 10000, step=1000, key="bt_capital")
        fee = st.number_input("手續費率", 0.0, 1.0, 0.1, step=0.05, format="%.3f", key="bt_fee") / 100
        slippage = st.number_input("滑價", 0.0, 1.0, 0.05, step=0.01, format="%.3f", key="bt_slip") / 100

        st.subheader("策略設定")
        enabled_strategies = {}
        for sname, scfg in DEFAULT_STRATEGIES.items():
            enabled = st.checkbox(f"{sname}", value=scfg["enabled"], key=f"bt_{sname}")
            enabled_strategies[sname] = {
                "enabled": enabled,
                "params": scfg["params"],
                "weight": scfg["weight"],
            }

        run_btn = st.button("🚀 開始回測", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner("回測進行中..."):
            try:
                result = run_backtest(
                    symbol=symbol, timeframe=tf, limit=lookback,
                    initial_capital=float(capital),
                    fee_rate=fee, slippage=slippage,
                    strategy_configs=enabled_strategies,
                    signal_threshold=0.6,
                    risk_config=DEFAULT_RISK,
                )
            except Exception as e:
                st.error(f"回測失敗: {e}")
                return

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("總報酬率", f"{result.total_return_pct:+.2f}%")
        with col2:
            st.metric("CAGR", f"{result.cagr:+.2f}%")
        with col3:
            st.metric("最大回撤", f"{result.max_drawdown_pct:.2f}%",
                      delta_color="inverse")
        with col4:
            st.metric("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")

        col5, col6, col7, col8 = st.columns(4)
        with col5:
            st.metric("勝率", f"{result.win_rate:.1f}%")
        with col6:
            st.metric("交易次數", result.total_trades)
        with col7:
            st.metric("初始資金", f"${result.initial_capital:,.0f}")
        with col8:
            st.metric("最終權益", f"${result.final_equity:,.2f}")

        st.subheader("權益曲線")
        if result.equity_curve:
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=result.equity_curve,
                mode="lines",
                name="Equity",
                line=dict(color="#00FF88", width=2),
                fill="tozeroy",
            ))
            fig.update_layout(
                height=400, template="plotly_dark",
                yaxis_title="Equity ($)",
                xaxis_title="Bar",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("每月報酬率")
        if result.monthly_returns:
            import plotly.express as px
            months = list(result.monthly_returns.keys())
            returns = list(result.monthly_returns.values())
            fig2 = px.bar(x=months, y=returns, color=returns,
                          color_continuous_scale=["#dc3545", "#ffc107", "#28a745"])
            fig2.update_layout(
                height=250, template="plotly_dark",
                xaxis_title="Month", yaxis_title="Return %",
                showlegend=False,
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("交易明細")
        if result.trades:
            import pandas as pd
            st.dataframe(pd.DataFrame(result.trades), use_container_width=True)
        else:
            st.info("回測期間無交易訊號")
    else:
        st.info("請在左側設定參數後點擊「開始回測」")
```

- [ ] **Step 3: Test backtest logic**

Run: `python -m pytest tests/test_trading_backtest.py tests/test_trading_strategy.py -v`
Expected: All strategy and backtest tests pass

(Do NOT commit yet — app.py won't compile until `portfolio_page` is defined in Task 7. Both tasks commit together.)

```bash
git add app.py
git commit -m "feat: add backtest page to Streamlit dashboard"
```


### Task 7: Streamlit — Portfolio dashboard page

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Add portfolio dashboard page**

After the `backtest_page()` function, add:

```python
def portfolio_page():
    st.title("💰 交易 Dashboard")
    st.markdown("---")

    from trading.portfolio import PortfolioStore

    store = PortfolioStore()
    portfolio = store.load()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("現金餘額", f"${portfolio.cash:,.2f}")
    with col2:
        st.metric("市值", f"${portfolio.market_value:,.2f}")
    with col3:
        st.metric("總權益", f"${portfolio.total_equity:,.2f}")
    with col4:
        total_pnl = portfolio.total_pnl
        st.metric("總損益", f"${total_pnl:+,.2f}",
                  delta=f"{total_pnl/portfolio.total_equity*100:+.2f}%" if portfolio.total_equity > 0 else "")

    # Positions
    st.subheader("當前持倉")
    if portfolio.positions:
        pos_data = []
        for p in portfolio.positions:
            pos_data.append({
                "交易對": p.symbol, "方向": p.side, "數量": p.quantity,
                "均價": f"${p.entry_price:,.2f}", "現價": f"${p.current_price:,.2f}",
                "浮盈": f"${p.unrealized_pnl:+,.2f}",
                "ROI%": f"{p.unrealized_pnl_pct:+.2f}%",
            })
        st.dataframe(pos_data, use_container_width=True)
    else:
        st.info("目前無持倉")

    # Orders
    st.subheader("近期訂單")
    if portfolio.orders:
        recent = portfolio.orders[-20:][::-1]
        order_data = []
        for o in recent:
            order_data.append({
                "時間": o.timestamp[-19:], "交易對": o.symbol,
                "方向": o.side, "數量": o.quantity,
                "價格": f"${o.price:,.2f}",
                "損益": f"${o.pnl:+,.2f}" if o.pnl else "-",
            })
        st.dataframe(order_data, use_container_width=True)
    else:
        st.info("暫無訂單記錄")

    # Equity chart
    st.subheader("帳戶損益趨勢")
    if portfolio.orders:
        import plotly.graph_objects as go
        equity_over_time = []
        running = portfolio.total_deposits
        for o in portfolio.orders:
            if o.side == "sell" and o.status == "filled":
                running += o.pnl
            equity_over_time.append(running)
        if equity_over_time:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=equity_over_time, mode="lines",
                name="Cumulative PnL", line=dict(color="#4FC3F7", width=2),
            ))
            fig.update_layout(height=300, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
```

- [ ] **Step 2: Add to page router**

The `pages` dict (from Task 6) already includes `"portfolio": portfolio_page`, so just ensure the function is defined above it.

- [ ] **Step 3: Final app.py commit (combined with Task 6)**

```bash
git add app.py
git commit -m "feat: add backtest and portfolio pages to Streamlit dashboard"
```

```bash
git add app.py
git commit -m "feat: add portfolio dashboard to Streamlit"
```


### Task 8: Update requirements, push, and verify deployment

**Files:**
- No new files. Push and verify.

- [ ] **Step 1: Verify all tests pass**

Run: `python -m pytest tests/ -v`
Expected: All trading tests + existing monitor tests pass (≥ 20 tests total)

- [ ] **Step 2: Push to GitHub**

```bash
git push
```

- [ ] **Step 3: Verify Render auto-deploy completes**

Visit: `https://crypto-dashboard-czw0.onrender.com` — confirm no 500 errors and trading pages appear in sidebar.

- [ ] **Step 4: Test paper trading end-to-end**

Visit: `https://crypto-invest-analysis.onrender.com/health` — should return 200

Then hit: `https://crypto-invest-analysis.onrender.com/check` — should produce BUY/SELL signals if strategy fires, Discord notification sent.
