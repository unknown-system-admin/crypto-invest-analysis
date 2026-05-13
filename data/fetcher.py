import pandas as pd
import ccxt


def fetch_ohlcv(symbol: str, timeframe: str = "1h", limit: int = 200) -> pd.DataFrame:
    exchange = ccxt.binance({"enableRateLimit": True})
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    except ccxt.BadSymbol:
        raise ValueError(f"Invalid symbol: {symbol}")
    except ccxt.NetworkError as e:
        raise ConnectionError(f"Network error fetching {symbol}: {e}")
    except ccxt.RateLimitExceeded as e:
        raise RuntimeError(f"Rate limit exceeded for {symbol}: {e}")
    except ccxt.ExchangeError as e:
        raise RuntimeError(f"Exchange error for {symbol}: {e}")

    df = pd.DataFrame(
        ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df
