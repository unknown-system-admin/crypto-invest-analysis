"""M1: Backtest hybrid signal on real OKX data; calibrate HTF threshold.

Not unit-tested: it is a research script. Run it and record results.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.signals import momentum_series
from data_cache import load_or_fetch

SYMBOLS = ["BTC/USDT:USDT", "SOL/USDT:USDT"]
HTF = "1h"
LTF = "15m"
FEE_RATE = 0.001  # per side
STOP_LOSS_PCT = 0.025
INITIAL = 10000.0


def run_backtest(df_htf, df_ltf, threshold, fee_rate=FEE_RATE, initial=INITIAL):
    htf_score = momentum_series(df_htf)
    ltf_score = momentum_series(df_ltf)
    ltf_delta = ltf_score.diff()

    ltf = df_ltf.copy()
    ltf["htf_score"] = htf_score.reindex(ltf.index, method="ffill")
    ltf["delta"] = ltf_delta.reindex(ltf.index)
    ltf["delta_prev"] = ltf["delta"].shift(1)
    ltf = ltf.dropna(subset=["htf_score", "delta", "delta_prev"])
    if len(ltf) == 0:
        return {"return_pct": 0.0, "trades": 0, "max_dd_pct": 0.0, "final": initial}

    equity = initial
    position = None  # "long"/"short"
    entry_price = None
    trades = 0
    peak = initial
    max_dd = 0.0

    for _, row in ltf.iterrows():
        price = row["close"]
        direction = ("long" if row["htf_score"] > threshold
                     else "short" if row["htf_score"] < -threshold else "flat")
        up = row["delta_prev"] <= 0 < row["delta"]
        down = row["delta_prev"] >= 0 > row["delta"]

        if position is None:
            if direction == "long" and up:
                position, entry_price, trades = "long", price, trades + 1
            elif direction == "short" and down:
                position, entry_price, trades = "short", price, trades + 1
        else:
            exit_now = (
                (position == "long" and (direction != "long" or down))
                or (position == "short" and (direction != "short" or up))
            )
            if position == "long" and (entry_price - price) / entry_price >= STOP_LOSS_PCT:
                exit_now = True
            if position == "short" and (price - entry_price) / entry_price >= STOP_LOSS_PCT:
                exit_now = True
            if exit_now:
                ret = (price / entry_price - 1) * (1 if position == "long" else -1)
                equity *= 1 + ret - 2 * fee_rate
                position, entry_price = None, None

        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak)

    return {
        "return_pct": (equity / initial - 1) * 100,
        "trades": trades,
        "max_dd_pct": max_dd * 100,
        "final": equity,
    }


def calibrate():
    out = {}
    for symbol in SYMBOLS:
        print(f"=== {symbol} ===")
        df_htf = load_or_fetch(symbol, HTF, limit=800)
        df_ltf = load_or_fetch(symbol, LTF, limit=1440)
        print(f"  1h: {len(df_htf)} bars {df_htf.index[0]} -> {df_htf.index[-1]}")
        print(f"  5m: {len(df_ltf)} bars {df_ltf.index[0]} -> {df_ltf.index[-1]}")

        best = None
        for threshold in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
            r = run_backtest(df_htf, df_ltf, threshold)
            print(f"  threshold={threshold:.2f}: ret {r['return_pct']:+.1f}% "
                  f"trades={r['trades']} maxDD {r['max_dd_pct']:.1f}%")
            if best is None or r["return_pct"] > best[1]["return_pct"]:
                best = (threshold, r)
        out[symbol] = {
            "best_threshold": best[0],
            "return_pct": best[1]["return_pct"],
            "trades": best[1]["trades"],
            "max_dd_pct": best[1]["max_dd_pct"],
        }
        print(f"  -> BEST threshold={best[0]:.2f} ret {best[1]['return_pct']:+.1f}%\n")

    with open(Path(__file__).parent / "calibration_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved bot/calibration_results.json")
    return out


if __name__ == "__main__":
    calibrate()