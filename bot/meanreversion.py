from dataclasses import dataclass

import pandas as pd

from feature_engine.indicators import compute_all_indicators


@dataclass
class Signal:
    action: str  # enter_long/enter_short/exit_long/exit_short/none
    rsi: float


def latest_rsi(df: pd.DataFrame) -> pd.Series:
    return compute_all_indicators(df)["RSI"]


def htf_trend_up(df_htf: pd.DataFrame) -> bool:
    ind = compute_all_indicators(df_htf)
    return bool(df_htf["close"].iloc[-1] > ind["SMA_50"].iloc[-1])


def htf_trend_down(df_htf: pd.DataFrame) -> bool:
    ind = compute_all_indicators(df_htf)
    return bool(df_htf["close"].iloc[-1] < ind["SMA_50"].iloc[-1])


def evaluate_meanrev(df_htf, df_ltf, oversold, overbought, side="flat",
                     exit_long_rsi=55.0, exit_short_rsi=45.0) -> Signal:
    rsi = latest_rsi(df_ltf)
    prev_rsi = float(rsi.iloc[-2])
    curr_rsi = float(rsi.iloc[-1])

    if side == "long":
        if curr_rsi >= exit_long_rsi:
            return Signal("exit_long", curr_rsi)
        return Signal("none", curr_rsi)
    if side == "short":
        if curr_rsi <= exit_short_rsi:
            return Signal("exit_short", curr_rsi)
        return Signal("none", curr_rsi)
    if htf_trend_up(df_htf) and prev_rsi > oversold and curr_rsi <= oversold:
        return Signal("enter_long", curr_rsi)
    if htf_trend_down(df_htf) and prev_rsi < overbought and curr_rsi >= overbought:
        return Signal("enter_short", curr_rsi)
    return Signal("none", curr_rsi)