"""Funding-rate contrarian bias + history helpers."""
import pandas as pd


def funding_bias(rate: float, threshold: float) -> str:
    """Contrarian funding bias. Positive funding = crowded longs -> short bias."""
    if rate > threshold:
        return "short"
    if rate < -threshold:
        return "long"
    return "neutral"


def funding_series_from_rows(rows: list) -> pd.Series:
    """Build a sorted funding-rate Series indexed by timestamp (ms)."""
    if not rows:
        return pd.Series(dtype=float)
    ts = pd.to_datetime([r["timestamp"] for r in rows], unit="ms")
    rates = [r.get("fundingRate") for r in rows]
    s = pd.Series(rates, index=ts)
    return s[~s.index.duplicated(keep="last")].sort_index().astype(float)


def fetch_funding_history(exchange, symbol: str) -> pd.Series:
    """Fetch funding history; return sorted Series indexed by timestamp."""
    rows = exchange.fetch_funding_rate_history(symbol)
    return funding_series_from_rows(rows)


def get_funding_exchange():
    import ccxt
    return ccxt.binance({"enableRateLimit": True})
