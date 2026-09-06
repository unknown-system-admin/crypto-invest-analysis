from dataclasses import dataclass

import pandas as pd

from feature_engine.indicators import compute_all_indicators
from feature_engine.momentum import momentum_score, momentum_delta


@dataclass
class Signal:
    action: str  # enter_long / enter_short / exit_long / exit_short / none
    htf_score: float
    ltf_delta: float


def momentum_series(df: pd.DataFrame) -> pd.Series:
    ind = compute_all_indicators(df)
    ind["close"] = df["close"]
    return momentum_score(ind)


def htf_direction_from_score(score: float, threshold: float) -> str:
    if score > threshold:
        return "long"
    if score < -threshold:
        return "short"
    return "flat"


def _up_cross(prev: float, curr: float) -> bool:
    return prev <= 0 < curr


def _down_cross(prev: float, curr: float) -> bool:
    return prev >= 0 > curr


def evaluate(df_htf: pd.DataFrame, df_ltf: pd.DataFrame,
             threshold: float, side: str = "flat") -> Signal:
    htf_score = momentum_series(df_htf).iloc[-1]
    deltas = momentum_delta(momentum_series(df_ltf))
    prev_delta, curr_delta = float(deltas.iloc[-2]), float(deltas.iloc[-1])
    direction = htf_direction_from_score(htf_score, threshold)

    if side == "long":
        if direction != "long" or _down_cross(prev_delta, curr_delta):
            return Signal("exit_long", htf_score, curr_delta)
        return Signal("none", htf_score, curr_delta)
    if side == "short":
        if direction != "short" or _up_cross(prev_delta, curr_delta):
            return Signal("exit_short", htf_score, curr_delta)
        return Signal("none", htf_score, curr_delta)
    if direction == "long" and _up_cross(prev_delta, curr_delta):
        return Signal("enter_long", htf_score, curr_delta)
    if direction == "short" and _down_cross(prev_delta, curr_delta):
        return Signal("enter_short", htf_score, curr_delta)
    return Signal("none", htf_score, curr_delta)