from data.fetcher import fetch_ohlcv
from indicators.calculator import compute_all
from analysis.summary import analyze_signals


TF_LABELS = {
    "1h": "1 小時",
    "4h": "4 小時",
    "1d": "1 日",
    "1w": "1 週",
}


def analyze_multi_timeframe(symbol: str) -> list:
    results = []
    for tf in ["1h", "4h", "1d", "1w"]:
        try:
            df = fetch_ohlcv(symbol, tf, limit=200)
        except Exception:
            results.append({"tf": tf, "label": TF_LABELS.get(tf, tf), "error": True})
            continue

        result = compute_all(df)
        overlay = result["overlay"]
        subplots = result["subplots"]
        close = overlay["close"].iloc[-1]
        sma20 = overlay["SMA_20"].iloc[-1]
        sma50 = overlay["SMA_50"].iloc[-1]
        rsi = subplots["rsi"]["RSI"].iloc[-1]
        macd_line = subplots["macd"]["MACD"].iloc[-1]
        signal = subplots["macd"]["Signal"].iloc[-1]
        sig = analyze_signals(overlay, subplots)
        bb_upper = overlay["BB_upper"].iloc[-1]
        bb_lower = overlay["BB_lower"].iloc[-1]

        results.append({
            "tf": tf,
            "label": TF_LABELS.get(tf, tf),
            "close": close,
            "sma20": sma20,
            "sma50": sma50,
            "rsi": rsi,
            "macd_bullish": bool(macd_line > signal),
            "bb_width": bb_upper - bb_lower,
            "direction": sig["direction"],
            "bullish": sig["bullish_count"],
            "bearish": sig["bearish_count"],
            "error": False,
        })
    return results
