#!/usr/bin/env python
"""Data caching system for OKX OHLCV data with CSV storage."""

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import ccxt
import pandas as pd


CACHE_DIR = Path(__file__).parent / "data_cache"


def _cache_path(symbol: str, timeframe: str) -> Path:
    """Generate cache file path for a given symbol/timeframe."""
    safe_symbol = symbol.replace("/", "_")
    return CACHE_DIR / f"{safe_symbol}_{timeframe}.csv"


def fetch_historical_data(
    symbol: str,
    timeframe: str,
    limit: int = 8000,
    since: int = None,
) -> pd.DataFrame:
    """Batch fetch OHLCV data from OKX with pagination.

    Fetches data backwards from the most recent candle.
    OKX returns at most 300 candles per request.
    Uses OKX's 'after' parameter to paginate to older data.
    """
    exchange = ccxt.okx({
        "apiKey": os.getenv("OKX_API_KEY"),
        "secret": os.getenv("OKX_API_SECRET"),
        "password": os.getenv("OKX_API_PASSPHRASE"),
        "enableRateLimit": True,
    })

    all_candles = []
    chunk_size = 300
    fetched = 0

    # Start with no 'since' to get most recent candles
    # Then paginate backwards using OKX's 'after' parameter
    after_ts = None

    while fetched < limit:
        batch_limit = min(chunk_size, limit - fetched)
        try:
            kwargs = {"symbol": symbol, "timeframe": timeframe, "limit": batch_limit}
            if after_ts is not None:
                kwargs["params"] = {"after": str(after_ts)}
            candles = exchange.fetch_ohlcv(**kwargs)
        except Exception as e:
            print(f"  Fetch error at offset {fetched}: {e}")
            break

        if not candles:
            break

        all_candles.extend(candles)
        fetched += len(candles)

        # Move backwards: set 'after' to earliest timestamp
        # This gives us the batch before the current batch
        earliest_ts = candles[0][0]
        after_ts = earliest_ts

        if len(candles) < batch_limit:
            break

    if not all_candles:
        return pd.DataFrame()

    df = pd.DataFrame(
        all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df[~df.index.duplicated(keep="first")]
    df.sort_index(inplace=True)
    return df


def save_to_cache(df: pd.DataFrame, symbol: str, timeframe: str) -> Path:
    """Save DataFrame to CSV cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(symbol, timeframe)
    df.to_csv(path)
    return path


def load_from_cache(symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
    """Load cached data from CSV. Returns None if no cache exists."""
    path = _cache_path(symbol, timeframe)
    if not path.exists():
        return None

    df = pd.read_csv(path, index_col="timestamp", parse_dates=True)
    return df


def update_cache(
    symbol: str,
    timeframe: str,
    limit: int = 8000,
) -> pd.DataFrame:
    """Update existing cache with new data only.

    Loads existing cache, fetches new candles since last cached timestamp,
    appends, and saves.
    """
    cached = load_from_cache(symbol, timeframe)

    if cached is not None and len(cached) > 0:
        last_ts = int(cached.index[-1].timestamp() * 1000)
        since_ms = last_ts + 1
        new_data = fetch_historical_data(symbol, timeframe, limit=limit, since=since_ms)

        if len(new_data) > 0:
            df = pd.concat([cached, new_data])
            df = df[~df.index.duplicated(keep="last")]
            df.sort_index(inplace=True)
        else:
            df = cached
    else:
        df = fetch_historical_data(symbol, timeframe, limit=limit)

    if len(df) > 0:
        save_to_cache(df, symbol, timeframe)

    return df


def load_or_fetch(
    symbol: str,
    timeframe: str,
    limit: int = 8000,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Load from cache if available, otherwise fetch from API.

    Args:
        symbol: Trading pair (e.g. "BTC/USDT")
        timeframe: Candle timeframe (e.g. "1h", "4h", "1d")
        limit: Number of candles to fetch
        force_refresh: If True, ignore cache and fetch fresh data
    """
    if not force_refresh:
        cached = load_from_cache(symbol, timeframe)
        if cached is not None and len(cached) > 0:
            print(f"  Loaded {len(cached)} candles from cache for {symbol} {timeframe}")
            return cached

    print(f"  Fetching {symbol} {timeframe} from OKX...")
    df = fetch_historical_data(symbol, timeframe, limit=limit)
    if len(df) > 0:
        save_to_cache(df, symbol, timeframe)
        print(f"  Saved {len(df)} candles to cache")
    return df


if __name__ == "__main__":
    df = load_or_fetch("BTC/USDT", "1h", limit=8000)
    print(f"Got {len(df)} rows: {df.index[0]} -> {df.index[-1]}")
