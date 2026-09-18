#!/usr/bin/env python
"""Data caching system for OKX OHLCV data with CSV storage."""

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import ccxt
import pandas as pd


CACHE_DIR = Path(__file__).parent / "data_cache"


def _cache_path(symbol: str, timeframe: str) -> Path:
    """Generate cache file path for a given symbol/timeframe."""
    safe_symbol = symbol.replace("/", "_")
    return CACHE_DIR / f"{safe_symbol}_{timeframe}.csv"


def _okx():
    return ccxt.okx({
        "apiKey": os.getenv("OKX_API_KEY"),
        "secret": os.getenv("OKX_API_SECRET"),
        "password": os.getenv("OKX_API_PASSPHRASE"),
        "enableRateLimit": True,
    })


def _candles_to_df(candles) -> pd.DataFrame:
    if not candles:
        return pd.DataFrame()
    df = pd.DataFrame(
        candles, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df = df[~df.index.duplicated(keep="last")]
    df.sort_index(inplace=True)
    return df


def fetch_recent_ohlcv(symbol: str, timeframe: str, limit: int = 300) -> pd.DataFrame:
    """Fetch the most recent candles (single request)."""
    exchange = _okx()
    candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=min(limit, 300))
    return _candles_to_df(candles)


def fetch_historical_data(
    symbol: str,
    timeframe: str,
    limit: int = 8000,
    since: int = None,
) -> pd.DataFrame:
    """Batch fetch OHLCV data from OKX with pagination.

    If `since` is set, page forward from that timestamp.
    Otherwise page backwards from the most recent candle.
    OKX returns at most 300 candles per request.
    """
    exchange = _okx()
    all_candles = []
    chunk_size = 300
    fetched = 0

    if since is not None:
        cursor = since
        while fetched < limit:
            batch_limit = min(chunk_size, limit - fetched)
            try:
                candles = exchange.fetch_ohlcv(
                    symbol, timeframe=timeframe, limit=batch_limit, since=cursor
                )
            except Exception as e:
                print(f"  Fetch error at since {cursor}: {e}")
                break
            if not candles:
                break
            all_candles.extend(candles)
            fetched += len(candles)
            last_ts = candles[-1][0]
            if last_ts <= cursor:
                break
            cursor = last_ts + 1
            if len(candles) < batch_limit:
                break
    else:
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
            after_ts = candles[0][0]
            if len(candles) < batch_limit:
                break

    return _candles_to_df(all_candles)


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
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df


def _merge(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    if a is None or len(a) == 0:
        return b
    if b is None or len(b) == 0:
        return a
    df = pd.concat([a, b])
    df = df[~df.index.duplicated(keep="last")]
    df.sort_index(inplace=True)
    return df


def update_cache(
    symbol: str,
    timeframe: str,
    limit: int = 8000,
) -> pd.DataFrame:
    """Update existing cache with new data only.

    Loads existing cache, fetches new candles since last cached timestamp,
    appends, and saves. Also overlays the most recent 300 bars so the last
    forming candle is not stale.
    """
    cached = load_from_cache(symbol, timeframe)
    recent = fetch_recent_ohlcv(symbol, timeframe, limit=300)

    if cached is not None and len(cached) > 0:
        last_ts = int(cached.index[-1].timestamp() * 1000)
        since_ms = last_ts - 1
        try:
            new_data = fetch_historical_data(symbol, timeframe, limit=limit, since=since_ms)
        except Exception as e:
            print(f"  Incremental fetch failed, using recent overlay: {e}")
            new_data = pd.DataFrame()
        df = _merge(_merge(cached, new_data), recent)
    else:
        hist = fetch_historical_data(symbol, timeframe, limit=limit)
        df = _merge(hist, recent)

    if len(df) > 0:
        save_to_cache(df, symbol, timeframe)
    return df


def load_or_fetch(
    symbol: str,
    timeframe: str,
    limit: int = 8000,
    force_refresh: bool = False,
    refresh_latest: bool = True,
) -> pd.DataFrame:
    """Load from cache and overlay the latest exchange candles.

    Live callers should keep refresh_latest=True so dates/prices are not
    frozen on a days-old CSV.
    """
    if force_refresh:
        print(f"  Fetching {symbol} {timeframe} from OKX (force)...")
        df = fetch_historical_data(symbol, timeframe, limit=limit)
        recent = fetch_recent_ohlcv(symbol, timeframe, limit=300)
        df = _merge(df, recent)
        if len(df) > 0:
            save_to_cache(df, symbol, timeframe)
        return df

    cached = load_from_cache(symbol, timeframe)
    if cached is not None and len(cached) > 0 and not refresh_latest:
        print(f"  Loaded {len(cached)} candles from cache for {symbol} {timeframe}")
        return cached

    if cached is None or len(cached) == 0:
        print(f"  Fetching {symbol} {timeframe} from OKX...")
        df = fetch_historical_data(symbol, timeframe, limit=limit)
        recent = fetch_recent_ohlcv(symbol, timeframe, limit=300)
        df = _merge(df, recent)
        if len(df) > 0:
            save_to_cache(df, symbol, timeframe)
            print(f"  Saved {len(df)} candles to cache")
        return df

    print(f"  Cache hit ({len(cached)}), overlaying latest {symbol} {timeframe}")
    recent = fetch_recent_ohlcv(symbol, timeframe, limit=300)
    df = _merge(cached, recent)
    if len(df) > 0:
        save_to_cache(df, symbol, timeframe)
    return df


if __name__ == "__main__":
    df = load_or_fetch("BTC/USDT", "1h", limit=8000)
    print(f"Got {len(df)} rows: {df.index[0]} -> {df.index[-1]}")
