from typing import Optional
from datetime import datetime, timezone
import uuid

from trading.portfolio import Portfolio, Position, Order, calculate_position_size

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
