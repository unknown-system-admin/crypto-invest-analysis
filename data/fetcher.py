import time
import os
from typing import Optional

import pandas as pd
import ccxt


_CACHE_TTL = 180  # seconds


class _TTLCache:
    def __init__(self, ttl: int):
        self.ttl = ttl
        self._store = {}

    def get(self, key: str) -> Optional[pd.DataFrame]:
        entry = self._store.get(key)
        if entry is None:
            return None
        if time.monotonic() - entry["ts"] > self.ttl:
            del self._store[key]
            return None
        return entry["df"]

    def set(self, key: str, df: pd.DataFrame):
        self._store[key] = {"df": df, "ts": time.monotonic()}


_cache = _TTLCache(_CACHE_TTL)


def fetch_ohlcv(symbol: str, timeframe: str = "1d", limit: int = 200) -> pd.DataFrame:
    cache_key = f"{symbol}:{timeframe}:{limit}"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached
    exchange = ccxt.okx({
        "apiKey": os.getenv("OKX_API_KEY"),
        "secret": os.getenv("OKX_API_SECRET"),
        "password": os.getenv("OKX_API_PASSPHRASE"),
        "enableRateLimit": True,
    })
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
    _cache.set(cache_key, df.copy())
    return df
