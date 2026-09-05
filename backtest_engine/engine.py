from dataclasses import dataclass, field
from typing import List
import pandas as pd
import numpy as np
from backtest_engine.strategy import Strategy, Signal


@dataclass
class Position:
    symbol: str
    side: str  # "long" or "short"
    quantity: float
    entry_price: float
    current_price: float = 0.0

    @property
    def unrealized_pnl(self) -> float:
        if self.side == "long":
            return (self.current_price - self.entry_price) * self.quantity
        return (self.entry_price - self.current_price) * self.quantity

    @property
    def equity_value(self) -> float:
        """Correct equity contribution: cash already excludes this position's cost."""
        if self.side == "long":
            return self.current_price * self.quantity
        # Short: margin (entry cost, already deducted from cash) + PnL
        return self.entry_price * self.quantity + self.unrealized_pnl


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
    trend_filter: bool = False
    min_holding_bars: int = 0
    cooldown_bars: int = 0

    def run(self, features: pd.DataFrame) -> BacktestResult:
        cash = self.initial_capital
        positions: List[Position] = []
        equity_curve = []
        trades = []
        daily_trade_count = 0
        current_day = None
        bars_held = 0
        bars_since_exit = None

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
            current_equity = cash + sum(p.equity_value for p in positions)
            drawdown = ((peak - current_equity) / peak * 100) if peak > 0 else 0

            if drawdown > self.max_drawdown_stop and positions:
                for pos in positions:
                    if pos.side == "long":
                        effective_price = price * (1 - self.slippage)
                        pnl = (effective_price * (1 - self.fee_rate) - pos.entry_price * (1 + self.fee_rate)) * pos.quantity
                        cash += pos.quantity * effective_price * (1 - self.fee_rate)
                    else:
                        effective_price = price * (1 + self.slippage)
                        pnl = (pos.entry_price * (1 - self.fee_rate) - effective_price * (1 + self.fee_rate)) * pos.quantity
                        # Return margin + PnL
                        margin_returned = pos.entry_price * pos.quantity
                        cash += margin_returned + pnl
                    trades.append({"action": "sell", "price": effective_price, "reason": "drawdown_stop", "pnl": pnl})
                positions.clear()

            # Execute trades
            if daily_trade_count < self.max_daily_trades:
                sma200 = row.get("SMA_200")

                if positions:
                    # Exit logic (min_holding gate; drawdown stop above is unaffected)
                    if positions[0].side == "long" and sig.direction == "偏空" and bars_held >= self.min_holding_bars:
                        pos = positions[0]
                        effective_price = price * (1 - self.slippage)
                        pnl = (effective_price * (1 - self.fee_rate) - pos.entry_price * (1 + self.fee_rate)) * pos.quantity
                        cash += pos.quantity * effective_price * (1 - self.fee_rate)
                        trades.append({"action": "sell", "price": effective_price, "quantity": pos.quantity, "pnl": pnl})
                        positions.clear()
                        daily_trade_count += 1
                        bars_held = 0
                        bars_since_exit = 0
                    elif positions[0].side == "short" and sig.direction == "偏多" and bars_held >= self.min_holding_bars:
                        pos = positions[0]
                        effective_price = price * (1 + self.slippage)
                        pnl = (pos.entry_price * (1 - self.fee_rate) - effective_price * (1 + self.fee_rate)) * pos.quantity
                        margin_returned = pos.entry_price * pos.quantity
                        cash += margin_returned + pnl
                        trades.append({"action": "cover", "price": effective_price, "quantity": pos.quantity, "pnl": pnl})
                        positions.clear()
                        daily_trade_count += 1
                        bars_held = 0
                        bars_since_exit = 0
                    else:
                        bars_held += 1
                else:
                    # Entry logic (trend filter + cooldown gate)
                    cooldown_ok = bars_since_exit is None or bars_since_exit >= self.cooldown_bars
                    trend_up = trend_down = True
                    if self.trend_filter:
                        if sma200 is None or pd.isna(sma200):
                            trend_up = trend_down = False
                        else:
                            trend_up = price > sma200
                            trend_down = price < sma200

                    if cooldown_ok and sig.direction == "偏多" and trend_up:
                        effective_price = price * (1 + self.slippage)
                        qty = (cash * self.max_position_pct / 100) / effective_price
                        if qty > 0:
                            cost = qty * effective_price * (1 + self.fee_rate)
                            if cost <= cash:
                                cash -= cost
                                positions.append(Position(self.symbol, "long", qty, effective_price, current_price=effective_price))
                                trades.append({"action": "buy", "price": effective_price, "quantity": qty})
                                daily_trade_count += 1
                                bars_held = 0
                    elif cooldown_ok and sig.direction == "偏空" and trend_down:
                        effective_price = price * (1 - self.slippage)
                        margin_required = cash * self.max_position_pct / 100
                        qty = margin_required / effective_price
                        if qty > 0 and margin_required <= cash and margin_required > 0:
                            cash -= margin_required
                            positions.append(Position(self.symbol, "short", qty, effective_price, current_price=effective_price))
                            trades.append({"action": "short_sell", "price": effective_price, "quantity": qty})
                            daily_trade_count += 1
                            bars_held = 0
                    if bars_since_exit is not None:
                        bars_since_exit += 1

            # Update equity
            unrealized = sum(p.equity_value for p in positions)
            current_equity = cash + unrealized
            equity_curve.append(current_equity)

        # Calculate metrics
        final_equity = equity_curve[-1] if equity_curve else cash
        total_return = ((final_equity - self.initial_capital) / self.initial_capital) * 100
        peak = max(equity_curve) if equity_curve else cash
        max_dd = ((peak - min(equity_curve)) / peak * 100) if equity_curve and peak > 0 else 0

        # Win rate: include both sells and covers
        exit_trades = [t for t in trades if t["action"] in ("sell", "cover") and "pnl" in t]
        wins = sum(1 for t in exit_trades if t["pnl"] > 0)
        total_exits = len(exit_trades)
        win_rate = (wins / total_exits * 100) if total_exits > 0 else 0

        # Sharpe ratio
        if len(equity_curve) > 1:
            returns = pd.Series(equity_curve).pct_change().dropna()
            if returns.std() > 0:
                periods_per_year = {
                    "1m": 60 * 24 * 365,
                    "5m": 12 * 24 * 365,
                    "15m": 4 * 24 * 365,
                    "1h": 24 * 365,
                    "4h": 6 * 365,
                    "1d": 365,
                    "1w": 52,
                }.get(self.timeframe, 24 * 365)
                sharpe = (returns.mean() / returns.std()) * np.sqrt(periods_per_year)
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0

        return BacktestResult(
            total_trades=len([t for t in trades if t["action"] in ("buy", "short_sell")]),
            final_equity=round(final_equity, 2),
            total_return_pct=round(total_return, 2),
            max_drawdown_pct=round(max_dd, 2),
            sharpe_ratio=round(sharpe, 2),
            win_rate=round(win_rate, 1),
            equity_curve=equity_curve,
            trades=trades,
        )
