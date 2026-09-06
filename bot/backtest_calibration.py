"""M1 v2: Backtest hybrid signal on Binance long history; calibrate params.

Grid: htf_threshold x stop mode (hard/trailing) x atr_stop_mult x min_atr_pct.
Requires network (Binance public API). Saves bot/calibration_results_v2.json.
"""
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.binance_data import fetch_ohlcv_long, get_binance_exchange
from bot.signals import momentum_series
from bot.risk import check_trailing_stop

SYMBOLS = ["BTC/USDT:USDT", "SOL/USDT:USDT"]
HTF, LTF = "1h", "15m"
HTF_LIMIT, LTF_LIMIT = 800, 8000
FEE_RATE = 0.001
HARD_STOP_PCT = 0.025
INITIAL = 10000.0

GRID = {
    "threshold": [0.10, 0.15, 0.20, 0.25, 0.30],
    "trailing": [False, True],
    "atr_mult": [2.0, 3.0, 4.0],
    "min_atr_pct": [0.0, 0.001, 0.002],
}


def latest_atr_series(df):
    from feature_engine.indicators import compute_all_indicators
    return compute_all_indicators(df)["ATR"]


def run_backtest(df_htf, df_ltf, threshold, trailing, atr_mult, min_atr_pct,
                 fee_rate=FEE_RATE, initial=INITIAL):
    htf_score = momentum_series(df_htf)
    ltf_score = momentum_series(df_ltf)
    ltf_delta = ltf_score.diff()
    atr = latest_atr_series(df_ltf)

    ltf = df_ltf.copy()
    ltf["htf_score"] = htf_score.reindex(ltf.index, method="ffill")
    ltf["delta"] = ltf_delta.reindex(ltf.index)
    ltf["delta_prev"] = ltf["delta"].shift(1)
    ltf["atr"] = atr.reindex(ltf.index)
    ltf["vol_ok"] = (ltf["atr"] / ltf["close"]) >= min_atr_pct
    ltf = ltf.dropna(subset=["htf_score", "delta", "delta_prev", "atr"])
    if len(ltf) == 0:
        return {"return_pct": 0.0, "trades": 0, "max_dd_pct": 0.0}

    equity = initial
    position = None
    entry_price = peak = None
    trades = 0
    curve_peak = initial
    max_dd = 0.0

    for _, row in ltf.iterrows():
        price = row["close"]
        direction = ("long" if row["htf_score"] > threshold
                     else "short" if row["htf_score"] < -threshold else "flat")
        up = row["delta_prev"] <= 0 < row["delta"]
        down = row["delta_prev"] >= 0 > row["delta"]

        if position is None:
            if row["vol_ok"] and direction == "long" and up:
                position, entry_price, peak = "long", price, price
                trades += 1
            elif row["vol_ok"] and direction == "short" and down:
                position, entry_price, peak = "short", price, price
                trades += 1
        else:
            if position == "long":
                peak = max(peak, price)
                stop_hit = check_trailing_stop(entry_price, peak, price, row["atr"],
                                               "long", atr_mult if trailing else 0.0,
                                               HARD_STOP_PCT)
            else:
                peak = min(peak, price)
                stop_hit = check_trailing_stop(entry_price, peak, price, row["atr"],
                                               "short", atr_mult if trailing else 0.0,
                                               HARD_STOP_PCT)
            exit_now = stop_hit or (
                (position == "long" and (direction != "long" or down))
                or (position == "short" and (direction != "short" or up)))
            if exit_now:
                ret = (price / entry_price - 1) * (1 if position == "long" else -1)
                equity *= 1 + ret - 2 * fee_rate
                position = entry_price = peak = None

        curve_peak = max(curve_peak, equity)
        max_dd = max(max_dd, (curve_peak - equity) / curve_peak)

    return {"return_pct": (equity / initial - 1) * 100, "trades": trades,
            "max_dd_pct": max_dd * 100}


def calibrate():
    ex = get_binance_exchange()
    out = {}
    for symbol in SYMBOLS:
        print(f"=== {symbol} ===")
        df_htf = fetch_ohlcv_long(ex, symbol, HTF, HTF_LIMIT)
        df_ltf = fetch_ohlcv_long(ex, symbol, LTF, LTF_LIMIT)
        print(f"  1h: {len(df_htf)} bars {df_htf.index[0]} -> {df_htf.index[-1]}")
        print(f"  15m: {len(df_ltf)} bars {df_ltf.index[0]} -> {df_ltf.index[-1]}")

        best = None
        n = 0
        for threshold, trailing, mult, atr_pct in itertools.product(
                GRID["threshold"], GRID["trailing"], GRID["atr_mult"], GRID["min_atr_pct"]):
            r = run_backtest(df_htf, df_ltf, threshold, trailing, mult, atr_pct)
            n += 1
            if best is None or r["return_pct"] > best[1]["return_pct"]:
                best = ((threshold, trailing, mult, atr_pct), r)
        print(f"  ({n} combos tested)")
        print(f"  BEST: thr={best[0][0]} trailing={best[0][1]} mult={best[0][2]} "
              f"atr_pct={best[0][3]} -> ret {best[1]['return_pct']:+.1f}% "
              f"trades={best[1]['trades']} maxDD {best[1]['max_dd_pct']:.1f}%")
        out[symbol] = {"params": best[0], **best[1]}

    with open(Path(__file__).parent / "calibration_results_v2.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved bot/calibration_results_v2.json")
    return out


if __name__ == "__main__":
    calibrate()