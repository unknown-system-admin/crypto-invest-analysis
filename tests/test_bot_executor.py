from bot.executor import OKXExecutor


class FakeExchange:
    def __init__(self):
        self.leverage_calls = []
        self.orders = []
        self._balance = {"total": {"USDT": 10000.0}}
        self._positions = []
        self._last = 100.0

    def set_sandbox_mode(self, flag):
        self.sandbox = flag

    def set_leverage(self, leverage, symbol, params=None):
        self.leverage_calls.append((leverage, symbol, params))

    def fetch_balance(self):
        return self._balance

    def fetch_positions(self):
        return self._positions

    def fetch_ticker(self, symbol):
        return {"last": self._last}

    def create_market_buy_order(self, symbol, amount, params=None):
        self.orders.append({"symbol": symbol, "amount": amount, "side": "buy", "params": params})
        return {"id": "o1", "symbol": symbol, "filled": amount, "price": self._last,
                "fees": [], "side": "buy"}

    def create_market_sell_order(self, symbol, amount, params=None):
        self.orders.append({"symbol": symbol, "amount": amount, "side": "sell", "params": params})
        return {"id": "o2", "symbol": symbol, "filled": amount, "price": self._last,
                "fees": [], "side": "sell"}


def test_executor_uses_sandbox_by_default():
    ex = FakeExchange()
    OKXExecutor(exchange=ex)
    assert ex.sandbox is True


def test_set_leverage_passes_isolated():
    ex = FakeExchange()
    e = OKXExecutor(exchange=ex)
    e.set_leverage("BTC/USDT:USDT", 5)
    assert ex.leverage_calls[-1] == (5, "BTC/USDT:USDT", {"mgnMode": "isolated"})


def test_open_long_converts_notional_to_amount():
    ex = FakeExchange()
    e = OKXExecutor(exchange=ex)
    e.open_long("BTC/USDT:USDT", 1000.0)  # price 100 -> amount 10
    assert ex.orders[-1]["side"] == "buy"
    assert abs(ex.orders[-1]["amount"] - 10.0) < 1e-9


def test_open_short_and_close_long():
    ex = FakeExchange()
    e = OKXExecutor(exchange=ex)
    e.open_short("BTC/USDT:USDT", 1000.0)
    assert ex.orders[-1]["side"] == "sell"
    # fake a short position
    ex._positions = [{"symbol": "BTC/USDT:USDT", "side": "short", "contracts": 10.0}]
    e.close_position("BTC/USDT:USDT", "short")
    assert ex.orders[-1]["side"] == "buy"
    assert ex.orders[-1]["params"] == {"reduceOnly": True}


def test_get_equity():
    ex = FakeExchange()
    e = OKXExecutor(exchange=ex)
    assert e.get_equity() == 10000.0