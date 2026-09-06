"""Long-history OHLCV fetch from Binance public API + CSV cache.

Calibration-only. The live bot still executes on OKX demo.
"""
import pandas as pd
from pathlib import Path

DEFAULT_CACHE_DIR = Path(__file__).parent / "cache_binance"
BATCH = 1000


def cache_path(symbol: str, timeframe: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    safe = symbol.replace("/", "_").replace(":", "_")
    return cache_dir / f"{safe}_{timeframe}.csv"


def fetch_ohlcv_long(exchange, symbol: str, timeframe: str, limit: int,
                     cache_dir: Path = DEFAULT_CACHE_DIR, force_refresh: bool = False) -> pd.DataFrame:
    path = cache_path(symbol, timeframe, cache_dir)
    if path.exists() and not force_refresh:
        df = pd.read_csv(path, index_col="timestamp", parse_dates=True)
        if len(df) >= limit:
            return df.tail(limit)

    interval_ms = {"5m": 300000, "15m": 900000, "1h": 3600000}[timeframe]
    since = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000) - (limit + 1) * interval_ms

    rows = []
    cursor = since
    while len(rows) < limit:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=BATCH)
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < BATCH:
            break
        cursor = batch[-1][0] + 1
        if len(rows) >= limit:
            break

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")
    df = df.set_index("timestamp").tail(limit)

    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return df


def get_binance_exchange():
    import ccxt
    return ccxt.binance({"enableRateLimit": True})
