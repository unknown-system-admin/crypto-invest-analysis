"""Live signal notifier.

Looks at the latest price and momentum (including the forming daily bar)
and the virtual position from the validated rule. Discord fires ONLY when
that state produces a new entry or exit. Hold is silent.
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
    "buy": ("🟢", "進場：開多", 0x00FF00),
    "short_sell": ("🔴", "進場：開空", 0xFF0000),
    "sell": ("⚪", "出場：平多", 0xAAAAAA),
    "cover": ("⚪", "出場：平空", 0xAAAAAA),
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


def _forming(idx) -> bool:
    return datetime.now(timezone.utc) < (_ts(idx) + timedelta(days=1))


def compute_latest_signal(features, buy=0.05, sell=-0.30, cooldown=3,
                          trend_filter=True, strong_filter=False,
                          min_holding=0, dd_stop=50, pos=95):
    """Replay rules on data through the latest bar. Signal only if that bar traded."""
    if features is None or len(features) == 0:
        return {"action": None, "verdict": "no_data"}

    bar = features.iloc[-1]
    idx = features.index[-1]
    mark = float(bar.get("close", 0) or 0)
    score = float(bar.get("momentum_score", 0) or 0)
    delta = float(bar.get("momentum_delta", 0) or 0)
    meta = {
        "date": _bar_date(idx),
        "price": mark,
        "score": score,
        "delta": delta,
        "last_bar_date": _bar_date(idx),
        "last_close": mark,
        "forming": _forming(idx),
        "bars": int(len(features)),
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
    result = engine.run(features)
    last = result.trades[-1] if result.trades else None
    fired = bool(last) and str(last.get("date", ""))[:10] == meta["date"]

    if not fired:
        side = "flat"
        since = None
        if last and last.get("action") in ("buy", "short_sell"):
            side = "long" if last["action"] == "buy" else "short"
            since = last.get("date")
        return {"action": None, "verdict": "hold", "position": side, "position_since": since, **meta}

    side = {"buy": "long", "short_sell": "short", "sell": "flat", "cover": "flat"}.get(last["action"], "flat")
    return {
        "action": last["action"],
        "verdict": "signal",
        "reason": last.get("reason"),
        "pnl": last.get("pnl"),
        "position": side,
        "price": mark,
        "score": score,
        **meta,
    }


def build_signal_embed(symbol: str, sig: dict) -> dict:
    emoji, label, color = ACTIONS.get(sig["action"], ("🔔", sig["action"], 0xAAAAAA))
    forming = "形成中" if sig.get("forming") else "已收盤"
    lines = [
        f"**{symbol}** {label}",
        f"💰 當下價格：${sig['price']:,.2f} · {sig['date']} UTC（{forming}）",
        f"📊 趨勢分數 {sig['score']:.3f} · 變化 {sig.get('delta', 0):+.3f}",
        f"門檻：進場 > {sig.get('buy_threshold', 0.05):.2f} · 出場 < {sig.get('sell_threshold', -0.30):.2f}",
    ]
    if sig.get("reason") == "drawdown_stop":
        lines.append("⚠️ 最大回撤停損")
    if sig.get("pnl") is not None:
        lines.append(f"📈 平倉損益：${sig['pnl']:+,.2f}")
    lines.append("\n👉 請自行至 OKX 下單（通知，非自動執行）")
    return {"embeds": [{"title": f"{emoji} {symbol} 策略訊號", "description": "\n".join(lines), "color": color}]}


def evaluate_symbol(symbol: str, features, send_fn, path=SIGNAL_STATE_PATH, cfg: dict = None) -> dict:
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
        "notified": False,
    }
    fresh = sig.get("action") and (
        sig.get("date") != prev.get("date") or sig.get("action") != prev.get("action")
    )
    if fresh:
        payload["notified"] = bool(send_fn(build_signal_embed(symbol, sig)))
        state[symbol] = {"date": sig["date"], "action": sig["action"], "price": sig["price"]}
        save_signal_state(state, path)
        return payload
    payload["reason"] = "already_notified" if sig.get("action") else "no_signal"
    return payload


def now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
