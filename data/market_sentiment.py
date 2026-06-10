import ccxt


def _perpetual_symbol(symbol: str) -> str:
    base = symbol.split("/")[0]
    return f"{base}/USDT:USDT"


def fetch_funding_rate(symbol: str) -> dict:
    exchange = ccxt.binance({"enableRateLimit": True})
    data = exchange.fetch_funding_rate(_perpetual_symbol(symbol))
    rate = data["fundingRate"]
    return {
        "rate": rate,
        "rate_pct": round(rate * 100, 4),
        "mark_price": data["markPrice"],
        "next_time": data["fundingDatetime"],
    }


def fetch_open_interest(symbol: str) -> dict:
    exchange = ccxt.binance({"enableRateLimit": True})
    data = exchange.fetch_open_interest(symbol)
    perp = _perpetual_symbol(symbol)
    try:
        history = exchange.fetch_open_interest_history(perp, timeframe="4h", limit=12)
        oi_12h_ago = history[0]["openInterestAmount"] if history else None
    except Exception:
        oi_12h_ago = None

    current = data["openInterestAmount"]
    if oi_12h_ago and oi_12h_ago > 0:
        oi_change_pct = round((current / oi_12h_ago - 1) * 100, 2)
    else:
        oi_change_pct = None

    return {
        "open_interest": current,
        "oi_change_pct": oi_change_pct,
    }


def fetch_orderbook_ratio(symbol: str, limit: int = 50) -> dict:
    exchange = ccxt.binance({"enableRateLimit": True})
    ob = exchange.fetch_order_book(symbol, limit=limit)
    bids_vol = sum(b[1] for b in ob["bids"])
    asks_vol = sum(a[1] for a in ob["asks"])
    ratio = bids_vol / asks_vol if asks_vol > 0 else 0
    best_bid = ob["bids"][0][0] if ob["bids"] else 0
    best_ask = ob["asks"][0][0] if ob["asks"] else 0
    spread = ((best_ask - best_bid) / best_bid) * 100 if best_bid > 0 else 0
    return {
        "bid_volume": round(bids_vol, 2),
        "ask_volume": round(asks_vol, 2),
        "bid_ask_ratio": round(ratio, 4),
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread_pct": round(spread, 4),
    }
