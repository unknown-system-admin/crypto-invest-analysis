import ccxt


class OKXExecutor:
    """OKX demo (sandbox) execution. Exchange injectable for tests."""

    def __init__(self, exchange=None):
        if exchange is None:
            exchange = ccxt.okx({
                "apiKey": _require_env("OKX_API_KEY"),
                "secret": _require_env("OKX_API_SECRET"),
                "password": _require_env("OKX_API_PASSPHRASE"),
                "enableRateLimit": True,
            })
        exchange.set_sandbox_mode(True)
        self.exchange = exchange

    def set_leverage(self, symbol: str, leverage: int) -> None:
        self.exchange.set_leverage(leverage, symbol, params={"mgnMode": "isolated"})

    def get_equity(self) -> float:
        bal = self.exchange.fetch_balance()
        return float(bal.get("total", {}).get("USDT", 0.0))

    def get_open_symbols(self) -> list:
        positions = self.exchange.fetch_positions()
        return [p["symbol"] for p in positions if float(p.get("contracts", 0) or 0) != 0]

    def _amount_from_notional(self, symbol: str, notional: float) -> float:
        price = self.exchange.fetch_ticker(symbol)["last"]
        if not price or price <= 0:
            raise ValueError(f"invalid ticker price for {symbol}")
        market = self.exchange.market(symbol)
        contract_size = float(market.get("contractSize") or 1.0)
        return notional / price / contract_size

    def open_long(self, symbol: str, notional: float):
        amount = self._amount_from_notional(symbol, notional)
        return self.exchange.create_market_buy_order(
            symbol, amount, params={"tdMode": "isolated"})

    def open_short(self, symbol: str, notional: float):
        amount = self._amount_from_notional(symbol, notional)
        return self.exchange.create_market_sell_order(
            symbol, amount, params={"tdMode": "isolated"})

    def close_position(self, symbol: str, side: str):
        positions = self.exchange.fetch_positions()
        pos = next((p for p in positions
                    if p["symbol"] == symbol and p["side"] == side
                    and float(p.get("contracts", 0) or 0) != 0), None)
        if pos is None:
            return None
        amount = abs(float(pos["contracts"]))
        if side == "long":
            return self.exchange.create_market_sell_order(
                symbol, amount, params={"reduceOnly": True, "tdMode": "isolated"})
        return self.exchange.create_market_buy_order(
            symbol, amount, params={"reduceOnly": True, "tdMode": "isolated"})


def _require_env(name: str) -> str:
    import os
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(f"Missing environment variable: {name}")
    return val