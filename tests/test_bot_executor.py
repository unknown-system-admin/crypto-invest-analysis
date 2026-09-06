from bot.executor import OKXExecutor


class FakeExchange:
    def __init__(self):
        self.leverage_calls = []
        self.orders = []
        self._balance = {"total": {"USDT": 10000.0}}
        self._positions = []
        self._last = 100.0
        self._market = {"contractSize": 1.0}

    def market(self, symbol):
        return self._market

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
    e.open_long("BTC/USDT:USDT", 1000.0)  # price 100, contractSize 1.0 -> amount 10
    assert ex.orders[-1]["side"] == "buy"
    assert abs(ex.orders[-1]["amount"] - 10.0) < 1e-9
    assert ex.orders[-1]["params"] == {"tdMode": "isolated"}


def test_open_long_btc_contract_size():
    ex = FakeExchange()
    ex._market = {"contractSize": 0.01}  # BTC/USDT:USDT sends sz in contracts
    e = OKXExecutor(exchange=ex)
    e.open_long("BTC/USDT:USDT", 1000.0)  # 10 base units / 0.01 = 1000 contracts
    assert abs(ex.orders[-1]["amount"] - 1000.0) < 1e-9


def test_open_long_rejects_none_ticker_price():
    ex = FakeExchange()
    e = OKXExecutor(exchange=ex)
    ex._last = None
    try:
        e.open_long("BTC/USDT:USDT", 1000.0)
    except ValueError:
        return
    assert False, "expected ValueError for None ticker price"


def test_open_short_and_close_long():
    ex = FakeExchange()
    e = OKXExecutor(exchange=ex)
    e.open_short("BTC/USDT:USDT", 1000.0)
    assert ex.orders[-1]["side"] == "sell"
    assert ex.orders[-1]["params"] == {"tdMode": "isolated"}
    # fake a short position
    ex._positions = [{"symbol": "BTC/USDT:USDT", "side": "short", "contracts": 10.0}]
    e.close_position("BTC/USDT:USDT", "short")
    assert ex.orders[-1]["side"] == "buy"
    assert ex.orders[-1]["params"] == {"reduceOnly": True, "tdMode": "isolated"}


def test_live_mode_forces_cache_refresh(monkeypatch):
    """--live must feed the loop a fetch_fn that forces a cache refresh, so
    signals/stops use fresh prices instead of a stale CSV from a prior run."""
    import sys
    import bot.run_bot as run_bot_mod

    seen = {"force_refresh": None}
    captured = {}

    class DummyExecutor:
        pass

    def fake_load_or_fetch(symbol, timeframe, limit, force_refresh=False):
        seen["force_refresh"] = force_refresh
        return None

    def fake_run_bot(cfg, executor, fetch_fn):
        captured["fetch_fn"] = fetch_fn

    monkeypatch.setattr(sys, "argv", ["run_bot", "--live"])
    monkeypatch.setattr(run_bot_mod, "OKXExecutor", DummyExecutor)
    monkeypatch.setattr(run_bot_mod, "load_or_fetch", fake_load_or_fetch)
    monkeypatch.setattr(run_bot_mod, "run_bot", fake_run_bot)

    run_bot_mod.main()

    assert captured["fetch_fn"] is not None
    captured["fetch_fn"]("BTC/USDT:USDT", "1h", 100)
    assert seen["force_refresh"] is True


def test_dry_run_uses_cached_data(monkeypatch):
    """--dry-run keeps passing load_or_fetch unchanged (cached signals ok)."""
    import sys
    import bot.run_bot as run_bot_mod

    captured = {}

    def fake_run_bot(cfg, executor, fetch_fn):
        captured["fetch_fn"] = fetch_fn

    monkeypatch.setattr(sys, "argv", ["run_bot", "--dry-run"])
    monkeypatch.setattr(run_bot_mod, "run_bot", fake_run_bot)

    run_bot_mod.main()

    assert captured["fetch_fn"] is run_bot_mod.load_or_fetch


def test_get_equity():
    ex = FakeExchange()
    e = OKXExecutor(exchange=ex)
    assert e.get_equity() == 10000.0