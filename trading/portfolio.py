from dataclasses import dataclass, asdict
from typing import Literal
from datetime import datetime
import json
from pathlib import Path


@dataclass
class Order:
    id: str
    symbol: str
    side: Literal["buy", "sell"]
    quantity: float
    price: float
    fee: float
    timestamp: str
    status: Literal["filled", "pending", "cancelled"]
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
    positions: list[Position]
    orders: list[Order]
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
