from dataclasses import dataclass, field
from typing import List
import pandas as pd
import numpy as np
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
    symbol: str = "BTC/USDT"
    max_drawdown_stop: float = 30.0
    timeframe: str = "1h"
    
    def run(self, features: pd.DataFrame) -> BacktestResult:
        cash = self.initial_capital
        positions: List[Position] = []
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
            
            # Reset daily trade count on new day
            row_day = idx.date()
            if current_day != row_day:
                current_day = row_day
                daily_trade_count = 0
            
            # Update current price for positions
            for pos in positions:
                pos.current_price = price
            
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
                # Determine annualization factor based on timeframe
                periods_per_year = {
                    "1m": 60 * 24 * 365,
                    "5m": 12 * 24 * 365,
                    "15m": 4 * 24 * 365,
                    "1h": 24 * 365,
                    "4h": 6 * 365,
                    "1d": 365,
                    "1w": 52,
                }.get(self.timeframe, 24 * 365)  # default hourly
                sharpe = (returns.mean() / returns.std()) * np.sqrt(periods_per_year)
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
