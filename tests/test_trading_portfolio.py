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


def test_portfolio_to_dict_roundtrip():
    pos = Position(symbol="BTC/USDT", side="long", quantity=0.5, entry_price=50000, current_price=51000)
    pf = Portfolio(cash=5000, positions=[pos], orders=[])
    data = pf.to_dict()
    assert data["cash"] == 5000
    assert len(data["positions"]) == 1
    pf2 = Portfolio.from_dict(data)
    assert pf2.cash == pf.cash
    assert pf2.positions[0].quantity == pf.positions[0].quantity
