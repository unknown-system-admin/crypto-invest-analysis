import pandas as pd
import numpy as np


def calc_pivot_points(high: float, low: float, close: float) -> dict:
    p = (high + low + close) / 3
    return {
        "Pivot": p,
        "R1": 2 * p - low,
        "R2": p + (high - low),
        "R3": high + 2 * (p - low),
        "S1": 2 * p - high,
        "S2": p - (high - low),
        "S3": low - 2 * (high - p),
    }


def calc_swing_levels(df: pd.DataFrame, window: int = 5) -> dict:
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)

    swing_highs = []
    swing_lows = []
    for i in range(window, n - window):
        if highs[i] == max(highs[i - window : i + window + 1]):
            swing_highs.append(highs[i])
        if lows[i] == min(lows[i - window : i + window + 1]):
            swing_lows.append(lows[i])

    recent = swing_highs[-3:] if swing_highs else []
    recent_lows = swing_lows[-3:] if swing_lows else []

    return {
        "recent_resistances": sorted(recent, reverse=True) if recent else [],
        "recent_supports": sorted(recent_lows) if recent_lows else [],
    }


def calc_key_ma_levels(overlay: pd.DataFrame) -> dict:
    levels = {}
    for col in ["SMA_20", "SMA_50", "SMA_200", "EMA_12", "EMA_26"]:
        if col in overlay.columns:
            val = overlay[col].iloc[-1]
            if pd.notna(val):
                levels[col] = round(val, 2)
    if "BB_upper" in overlay.columns:
        levels["BB_upper"] = round(overlay["BB_upper"].iloc[-1], 2)
    if "BB_lower" in overlay.columns:
        levels["BB_lower"] = round(overlay["BB_lower"].iloc[-1], 2)
    if "BB_middle" in overlay.columns:
        levels["BB_middle"] = round(overlay["BB_middle"].iloc[-1], 2)
    return levels


def compute_all_levels(df: pd.DataFrame, overlay: pd.DataFrame) -> dict:
    latest = df.iloc[-1]
    pivot = calc_pivot_points(latest["high"], latest["low"], latest["close"])
    swings = calc_swing_levels(df)
    ma_levels = calc_key_ma_levels(overlay)
    return {
        "pivot_points": pivot,
        "swing_levels": swings,
        "ma_levels": ma_levels,
    }
