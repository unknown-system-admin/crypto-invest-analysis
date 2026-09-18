"""Strategy signal notification bot.

Replays the validated momentum strategy over daily data and notifies via
Discord ONLY when a new entry/exit signal fires (deduplicated by trade date).
Tracks a virtual position per symbol so signals are state-consistent with
the backtest. This is a NOTIFICATION bot, not an execution bot — the user
places orders manually on OKX.
"""

import json
from pathlib import Path
from datetime import datetime, timezone

from backtest_engine.engine import BacktestEngine
from backtest_engine.rule_strategy import MomentumRuleStrategy

SIGNAL_STATE_PATH = Path(__file__).parent / "signal_state.json"

# Validated config (see README strategy section) — frozen, do not re-tune.
DEFAULT_CFG = dict(
    buy=0.05,
    sell=-0.30,
    cooldown=3,
    trend_filter=True,
    strong_filter=False,
    min_holding=0,
    dd_stop=50,
    pos=95,
)

ACTIONS = {
    "buy": ("🟢", "買入訊號（開多單）", 0x00FF00),
    "short_sell": ("🔴", "放空訊號（開空單）", 0xFF0000),
    "sell": ("⚪", "平多訊號（出場）", 0xAAAAAA),
    "cover": ("⚪", "平空訊號（出場）", 0xAAAAAA),
}


def load_signal_state(path: Path = SIGNAL_STATE_PATH) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def save_signal_state(data: dict, path: Path = SIGNAL_STATE_PATH) -> None:
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _bar_date(idx) -> str:
    ts = pd_timestamp(idx)
    return ts.strftime("%Y-%m-%d")


def pd_timestamp(idx):
    import pandas as pd
    ts = pd.Timestamp(idx)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _is_forming_daily(idx) -> bool:
    """True if this 1d candle has not reached the next UTC midnight."""
    ts = pd_timestamp(idx)
    now = datetime.now(timezone.utc)
    return now < (ts + pd_timedelta_days(1))


def pd_timedelta_days(n: int):
    from datetime import timedelta
    return timedelta(days=n)


def compute_latest_signal(features, buy=0.05, sell=-0.30, cooldown=3,
                          trend_filter=True, strong_filter=False,
                          min_holding=0, dd_stop=50, pos=95):
    """Replay the strategy over `features`; return the most recent trade.

    Also attaches last-bar metadata so callers can distinguish signal-bar
    fill from the latest close.
    """
    last_bar = features.iloc[-1]
    last_idx = features.index[-1]
    last_close = float(last_bar.get("close", 0) or 0)
    last_score = float(last_bar.get("momentum_score", 0) or 0)
    meta = {
        "last_bar_date": _bar_date(last_idx),
        "last_close": last_close,
        "last_score": last_score,
        "forming": _is_forming_daily(last_idx),
        "bars": int(len(features)),
    }

    engine = BacktestEngine(
        strategy=MomentumRuleStrategy(buy_threshold=buy, sell_threshold=sell),
        initial_capital=10000,
        timeframe="1d",
        max_position_pct=pos,
        max_drawdown_stop=dd_stop,
        trend_filter=trend_filter,
        strong_filter=strong_filter,
        min_holding_bars=min_holding,
        cooldown_bars=cooldown,
    )
    result = engine.run(features)
    if not result.trades:
        return {"action": None, **meta}

    last = result.trades[-1]
    return {
        "action": last["action"],
        "date": last.get("date", ""),
        "price": last["price"],
        "score": last.get("score", 0),
        "reason": last.get("reason"),
        "pnl": last.get("pnl"),
        **meta,
    }


def build_signal_embed(symbol: str, sig: dict) -> dict:
    emoji, label, color = ACTIONS.get(sig["action"], ("🔔", sig["action"], 0xAAAAAA))
    lines = [f"**{symbol}** {label}"]
    lines.append(f"📅 訊號K（UTC）：{sig['date']}")
    lines.append(f"💰 訊號進場價：${sig['price']:,.2f}")
    if sig.get("last_close") is not None:
        forming = "（形成中）" if sig.get("forming") else "（已收盤）"
        lines.append(f"📊 最新收盤{forming}：${sig['last_close']:,.2f} · {sig.get('last_bar_date', '')}")
    lines.append(f"📊 訊號當根動能：{sig['score']:.3f}")
    if sig.get("reason") == "drawdown_stop":
        lines.append("⚠️ 因最大回撤停損出場")
    if sig.get("pnl") is not None:
        lines.append(f"📈 平倉損益：${sig['pnl']:+,.2f}")
    lines.append("\n👉 請自行至 OKX 下單（此為通知，非自動執行）")
    return {"embeds": [{"title": f"{emoji} {symbol} 策略訊號", "description": "\n".join(lines), "color": color}]}


def evaluate_symbol(symbol: str, features, send_fn, path=SIGNAL_STATE_PATH,
                    cfg: dict = None) -> dict:
    """Evaluate one symbol, notify on NEW signal, update state. Returns result dict."""
    cfg = cfg or DEFAULT_CFG
    sig = compute_latest_signal(
        features,
        buy=cfg["buy"], sell=cfg["sell"], cooldown=cfg["cooldown"],
        trend_filter=cfg["trend_filter"], strong_filter=cfg["strong_filter"],
        min_holding=cfg["min_holding"], dd_stop=cfg["dd_stop"], pos=cfg["pos"],
    )

    state = load_signal_state(path)
    prev = state.get(symbol, {})
    prev_date = prev.get("date", "")
    prev_action = prev.get("action", "")

    base = {
        "symbol": symbol,
        "last_bar_date": sig.get("last_bar_date"),
        "last_close": sig.get("last_close"),
        "forming": sig.get("forming"),
        "bars": sig.get("bars"),
    }

    if not sig.get("action"):
        return {**base, "signal": "none", "notified": False}

    new_signal = sig["date"] > prev_date and sig["action"] != prev_action

    payload = {
        **base,
        "signal": sig["action"],
        "date": sig["date"],
        "price": sig["price"],
        "score": sig.get("score"),
    }

    if new_signal:
        embed = build_signal_embed(symbol, sig)
        ok = send_fn(embed)
        state[symbol] = {"date": sig["date"], "action": sig["action"], "price": sig["price"]}
        save_signal_state(state, path)
        payload["notified"] = ok
        return payload

    payload["notified"] = False
    payload["reason"] = "already_notified"
    return payload


def now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
