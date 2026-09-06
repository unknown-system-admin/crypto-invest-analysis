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


def compute_latest_signal(features, buy=0.05, sell=-0.30, cooldown=3,
                          trend_filter=True, strong_filter=False,
                          min_holding=0, dd_stop=50, pos=95):
    """Replay the strategy over `features`; return the most recent trade.

    Returns None if no trades occurred, or a dict:
        {action, date, price, score, reason, pnl}
    """
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
        return None
    last = result.trades[-1]
    return {
        "action": last["action"],
        "date": last.get("date", ""),
        "price": last["price"],
        "score": last.get("score", 0),
        "reason": last.get("reason"),
        "pnl": last.get("pnl"),
    }


def build_signal_embed(symbol: str, sig: dict) -> dict:
    emoji, label, color = ACTIONS.get(sig["action"], ("🔔", sig["action"], 0xAAAAAA))
    lines = [f"**{symbol}** {label}"]
    lines.append(f"📅 日期：{sig['date']}")
    lines.append(f"💰 價格：${sig['price']:,.2f}")
    lines.append(f"📊 動能分數：{sig['score']:.3f}")
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

    if sig is None:
        return {"symbol": symbol, "signal": "none", "notified": False}

    new_signal = sig["date"] > prev_date and sig["action"] != prev_action

    if new_signal:
        embed = build_signal_embed(symbol, sig)
        ok = send_fn(embed)
        state[symbol] = {"date": sig["date"], "action": sig["action"], "price": sig["price"]}
        save_signal_state(state, path)
        return {"symbol": symbol, "signal": sig["action"], "date": sig["date"],
                "notified": ok, "price": sig["price"]}

    return {"symbol": symbol, "signal": sig["action"], "date": sig["date"],
            "notified": False, "reason": "already_notified"}


def now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")