from trading.portfolio import Portfolio, Position
from trading.executor import PaperExecutor, LiveExecutor

def test_paper_buy():
    pf = Portfolio(cash=10000, positions=[], orders=[])
    ex = PaperExecutor(pf)
    order = ex.execute_buy("BTC/USDT", 50000, 0.05)
    assert order.side == "buy"
    assert order.status == "filled"
    assert len(pf.positions) == 1
    assert pf.positions[0].quantity == 0.05
    assert pf.cash < 10000
    assert order.fee > 0

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
    assert round(pf.positions[0].quantity, 6) == 0.06

def test_can_trade_limit():
    pf = Portfolio(cash=10000, positions=[], orders=[])
    ex = PaperExecutor(pf)
    assert ex.can_trade("BTC/USDT", 9, 10) == True
    assert ex.can_trade("BTC/USDT", 10, 10) == False

def test_paper_buy_accumulates_position():
    pf = Portfolio(cash=10000, positions=[], orders=[])
    ex = PaperExecutor(pf)
    ex.execute_buy("BTC/USDT", 50000, 0.05)
    ex.execute_buy("BTC/USDT", 51000, 0.05)
    assert len(pf.positions) == 1
    assert pf.positions[0].quantity == 0.1
    assert pf.positions[0].entry_price > 50000  # weighted average

def test_live_executor_buy_no_exchange():
    """LiveExecutor without exchange falls back to paper"""
    pf = Portfolio(cash=10000, positions=[], orders=[])
    ex = LiveExecutor(pf)
    order = ex.execute_buy("BTC/USDT", 50000, 0.05)
    assert order.side == "buy"
    assert order.status == "filled"
    assert len(pf.positions) == 1
    assert pf.cash < 10000

def test_live_executor_sell_no_exchange():
    pf = Portfolio(cash=10000, positions=[Position(symbol="BTC/USDT", side="long", quantity=0.05, entry_price=50000, current_price=50000)], orders=[])
    ex = LiveExecutor(pf)
    order = ex.execute_sell("BTC/USDT", 55000, 0.05)
    assert order is not None
    assert order.side == "sell"
    assert order.pnl > 0
    assert len(pf.positions) == 0
