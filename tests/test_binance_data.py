import time

import pandas as pd

from bot.binance_data import cache_path, fetch_ohlcv_long

INTERVAL_MS = 900000
LIMIT = 1200


class FakeBinance:
    """Binance double: candles span [now - LIMIT*interval, now] like live data."""

    def __init__(self):
        # Window extends 3x past `since` so the impl always collects >= limit
        # rows regardless of ms drift; .tail(limit) then trims to exactly 1200.
        end_ms = int(time.time() * 1000)
        self._rows = [
            [end_ms - (3 * LIMIT - i) * INTERVAL_MS, 100.0, 101.0, 99.0, 100.5, 1000.0]
            for i in range(3 * LIMIT)
        ]

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None):
        start = 0
        if since is not None:
            start = max(0, (since - self._rows[0][0] + INTERVAL_MS - 1) // INTERVAL_MS)
        return self._rows[start:start + 1000]


def test_cache_path_sanitizes_symbol():
    p = cache_path("BTC/USDT:USDT", "15m")
    assert ":" not in p.name and "/" not in p.name


def test_fetch_ohlcv_long_paginates(tmp_path):
    fake = FakeBinance()
    df = fetch_ohlcv_long(fake, "BTC/USDT:USDT", "15m", limit=LIMIT,
                          cache_dir=tmp_path)
    assert len(df) == LIMIT
    assert df["close"].iloc[-1] == 100.5
    assert isinstance(df.index, pd.DatetimeIndex)


def test_fetch_ohlcv_long_uses_cache(tmp_path):
    fake = FakeBinance()
    df1 = fetch_ohlcv_long(fake, "BTC/USDT:USDT", "15m", limit=LIMIT,
                           cache_dir=tmp_path)
    fake._rows = []  # if cache used, no refetch -> still returns data
    df2 = fetch_ohlcv_long(fake, "BTC/USDT:USDT", "15m", limit=LIMIT,
                           cache_dir=tmp_path)
    assert len(df2) == LIMIT
