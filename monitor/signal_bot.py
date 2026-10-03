"""Strategy signal notification bot.

Replays the validated momentum strategy, then notifies only when the
latest CLOSED daily bar itself produced an entry/exit. Historical opens
are position context, not a fresh buy/sell at today's price.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from backtest_engine.engine import BacktestEngine
from backtest_engine.rule_strategy import MomentumRuleStrategy

SIGNAL_STATE_PATH = Path(__file__).parent / "signal_state.json"

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
    "buy": ("🟢", "買入訊號（現價開多）", 0x00FF00),
    "short_sell": ("🔴", "放空訊號（現價開空）", 0xFF0000),
    "sell": ("⚪", "賣出訊號（現價平多）", 0xAAAAAA),
    "cover": ("⚪", "回補訊號（現價平空）", 0xAAAAAA),
}


def load_signal_state(path: Path = SIGNAL_STATE_PATH) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def save_signal_state(data: dict, path: Path = SIGNAL_STATE_PATH) -> None:
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _ts(idx) -> pd.Timestamp:
    ts = pd.Timestamp(idx)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _bar_date(idx) -> str:
    return _ts(idx).strftime("%Y-%m-%d")


def _is_forming_daily(idx) -> bool:
    return datetime.now(timezone.utc) < (_ts(idx) + timedelta(days=1))


def _decision_frame(features: pd.DataFrame):
    """Drop the still-forming daily bar so a signal uses a closed close."""
    if len(features) == 0:
        return features, None, True
    last_idx = features.index[-1]
    forming = _is_forming_daily(last_idx)
    if forming and len(features) > 1:
        return features.iloc[:-1], last_idx, True
    return features, last_idx, forming


def compute_latest_signal(features, buy=0.05, sell=-0.30, cooldown=3,
                          trend_filter=True, strong_filter=False,
                          min_holding=0, dd_stop=50, pos=95):
    """Judge the latest closed bar only.

    action is set only when the engine's last trade date equals that bar.
    price is that bar's close, not an older entry fill.
    """
    decision, live_idx, forming = _decision_frame(features)
    if decision is None or len(decision) == 0:
        return {"action": None, "verdict": "no_data"}

    bar = decision.iloc[-1]
    bar_idx = decision.index[-1]
    mark = float(bar.get("close", 0) or 0)
    score = float(bar.get("momentum_score", 0) or 0)
    delta = float(bar.get("momentum_delta", 0) or 0)
    live_close = None
    if live_idx is not None and forming:
        live_close = float(features.loc[live_idx].get("close", 0) or 0)

    meta = {
        "date": _bar_date(bar_idx),
        "price": mark,
        "score": score,
        "delta": delta,
        "last_bar_date": _bar_date(live_idx if live_idx is not None else bar_idx),
        "last_close": live_close if live_close is not None else mark,
        "forming": forming,
        "bars": int(len(decision)),
        "buy_threshold": buy,
        "sell_threshold": sell,
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
    result = engine.run(decision)
    last = result.trades[-1] if result.trades else None
    fired = bool(last) and last.get("date") == meta["date"]

    if not fired:
        side = "flat"
        if last and last.get("action") in ("buy", "short_sell"):
            side = "long" if last["action"] == "buy" else "short"
        return {
            "action": None,
            "verdict": "hold",
            "position": side,
            "position_since": None if side == "flat" else last.get("date"),
            **meta,
        }

    return {
        "action": last["action"],
        "verdict": "signal",
        "reason": last.get("reason"),
        "pnl": last.get("pnl"),
        "position": {"buy": "long", "short_sell": "short", "sell": "flat", "cover": "flat"}.get(last["action"], "flat"),
        **meta,
    }


def build_signal_embed(symbol: str, sig: dict) -> dict:
    emoji, label, color = ACTIONS.get(sig["action"], ("🔔", sig["action"], 0xAAAAAA))
    lines = [f"**{symbol}** {label}"]
    lines.append(f"📅 判斷K（已收盤 UTC）：{sig['date']}")
    lines.append(f"💰 判斷價（該根收盤）：${sig['price']:,.2f}")
    lines.append(
        f"📊 動能 {sig['score']:.3f} · delta {sig.get('delta', 0):+.3f}"
        f"（買>{sig.get('buy_threshold', 0.05):.2f} / 賣<{sig.get('sell_threshold', -0.30):.2f}）"
    )
    if sig.get("forming") and sig.get("last_close") is not None:
        lines.append(f"📊 形成中現價：${sig['last_close']:,.2f} · {sig.get('last_bar_date', '')}（不用於判斷）")
    if sig.get("reason") == "drawdown_stop":
        lines.append("⚠️ 因最大回撤停損出場")
    if sig.get("pnl") is not None:
        lines.append(f"📈 平倉損益：${sig['pnl']:+,.2f}")
    lines.append("\n👉 請自行至 OKX 下單（此為通知，非自動執行）")
    return {"embeds": [{"title": f"{emoji} {symbol} 策略訊號", "description": "\n".join(lines), "color": color}]}


def evaluate_symbol(symbol: str, features, send_fn, path=SIGNAL_STATE_PATH,
                    cfg: dict = None) -> dict:
    cfg = cfg or DEFAULT_CFG
    sig = compute_latest_signal(
        features,
        buy=cfg["buy"], sell=cfg["sell"], cooldown=cfg["cooldown"],
        trend_filter=cfg["trend_filter"], strong_filter=cfg["strong_filter"],
        min_holding=cfg["min_holding"], dd_stop=cfg["dd_stop"], pos=cfg["pos"],
    )
    state = load_signal_state(path)
    prev = state.get(symbol, {})
    payload = {
        "symbol": symbol,
        "verdict": sig.get("verdict"),
        "signal": sig.get("action") or "hold",
        "date": sig.get("date"),
        "price": sig.get("price"),
        "score": sig.get("score"),
        "delta": sig.get("delta"),
        "position": sig.get("position"),
        "position_since": sig.get("position_since"),
        "last_bar_date": sig.get("last_bar_date"),
        "last_close": sig.get("last_close"),
        "forming": sig.get("forming"),
        "bars": sig.get("bars"),
        "notified": False,
    }
    if sig.get("action") and sig.get("date") and (
        sig["date"] != prev.get("date") or sig["action"] != prev.get("action")
    ):
        ok = send_fn(build_signal_embed(symbol, sig))
        state[symbol] = {"date": sig["date"], "action": sig["action"], "price": sig["price"]}
        save_signal_state(state, path)
        payload["notified"] = ok
        return payload
    if sig.get("action"):
        payload["reason"] = "already_notified"
    else:
        payload["reason"] = "no_signal_on_closed_bar"
    return payload


def now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
