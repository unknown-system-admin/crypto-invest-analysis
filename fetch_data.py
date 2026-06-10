import requests
from datetime import datetime


BINANCE_API = "https://api.binance.com/api/v3/ticker/price"
SYMBOLS = ["BTCUSDT", "SOLUSDT"]


def fetch_prices(symbols: list[str]) -> dict[str, float]:
    prices = {}
    for symbol in symbols:
        resp = requests.get(BINANCE_API, params={"symbol": symbol})
        resp.raise_for_status()
        data = resp.json()
        prices[symbol] = float(data["price"])
    return prices


def format_symbol(symbol: str) -> str:
    return f"{symbol[:-4]}/{symbol[-4:]}"


def main():
    print(f"[{datetime.now().isoformat()}] Fetching prices from Binance...\n")
    try:
        prices = fetch_prices(SYMBOLS)
        for symbol, price in prices.items():
            print(f"  {format_symbol(symbol):>8}: ${price:,.2f}")
    except requests.RequestException as e:
        print(f"Error fetching prices: {e}")
        return
    print()


if __name__ == "__main__":
    main()
