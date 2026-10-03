"""Live signal notifier.

Price is the exchange last trade, not the daily close and not a historical fill.
No virtual position. Regime:
  long  if score > buy and delta > 0
  short if score < sell and delta < 0
  flat  otherwise
Discord fires on a new long/short and when an active regime disappears.
Flat with no prior regime is silent.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import ccxt

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
    "long": ("\U0001f7e2", "\u505a\u591a\u8a0a\u865f", 0x00FF00),
    "short": ("\U0001f534", "\u505a\u7a7a\u8a0a\u865f", 0xFF0000),
    "flat": ("\u26aa", "\u8da8\u52e2\u6d88\u5931", 0xAAAAAA),
}


def load_signal_state(path: Path = SIGNAL_STATE_PATH) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def save_signal_state(data: dict, path: Path = SIGNAL_STATE_PATH) -> None:
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def fetch_last_price(symbol: str) -> float:
    exchange = ccxt.okx({
        "apiKey": os.getenv("OKX_API_KEY"),
        "secret": os.getenv("OKX_API_SECRET"),
        "password": os.getenv("OKX_API_PASSPHRASE"),
        "enableRateLimit": True,
    })
    ticker = exchange.fetch_ticker(symbol)
    last = ticker.get("last") or ticker.get("close")
    if last is None:
        raise RuntimeError(f"no last price for {symbol}")
    return float(last)


def _regime(score: float, delta: float, buy: float, sell: float):
    if score > buy and delta > 0:
        return "long"
    if score < sell and delta < 0:
        return "short"
    return "flat"


def compute_latest_signal(features, symbol: str, buy=0.05, sell=-0.30, **_ignored):
    if features is None or len(features) == 0:
        return {"action": None, "verdict": "no_data"}
    bar = features.iloc[-1]
    score = float(bar.get("momentum_score", 0) or 0)
    delta = float(bar.get("momentum_delta", 0) or 0)
    price = fetch_last_price(symbol)
    regime = _regime(score, delta, buy, sell)
    return {
        "action": regime,
        "verdict": "signal" if regime != "flat" else "flat",
        "price": price,
        "score": score,
        "delta": delta,
        "buy_threshold": buy,
        "sell_threshold": sell,
        "candle_close": float(bar.get("close", 0) or 0),
    }


def build_signal_embed(symbol: str, sig: dict, previous: str = None) -> dict:
    emoji, label, color = ACTIONS.get(sig["action"], ("\U0001f514", sig["action"], 0xAAAAAA))
    lines = [
        f"**{symbol}** {label}",
        f"\U0001f4b0 \u7576\u4e0b\u6700\u65b0\u6210\u4ea4\u50f9\uff1a${sig['price']:,.2f}",
        f"\U0001f4ca \u8da8\u52e2\u5206\u6578 {sig['score']:.3f} \u00b7 \u8b8a\u5316 {sig.get('delta', 0):+.3f}",
        f"\u9580\u6abb\uff1a\u505a\u591a > {sig.get('buy_threshold', 0.05):.2f} \u4e14\u8b8a\u5316>0 \u00b7 \u505a\u7a7a < {sig.get('sell_threshold', -0.30):.2f} \u4e14\u8b8a\u5316<0",
    ]
    if sig["action"] == "flat" and previous:
        lines.append(f"\u5148\u524d\u8da8\u52e2\uff1a{previous} \u2192 \u5df2\u843d\u51fa\u9580\u6abb")
    lines.append("\n\U0001f449 \u8acb\u81ea\u884c\u81f3 OKX \u4e0b\u55ae\uff08\u901a\u77e5\uff0c\u975e\u81ea\u52d5\u57f7\u884c\uff09")
    return {"embeds": [{"title": f"{emoji} {symbol} \u7b56\u7565\u8a0a\u865f", "description": "\n".join(lines), "color": color}]}


def evaluate_symbol(symbol: str, features, send_fn, path=SIGNAL_STATE_PATH, cfg: dict = None) -> dict:
    cfg = cfg or DEFAULT_CFG
    sig = compute_latest_signal(features, symbol, buy=cfg["buy"], sell=cfg["sell"])
    state = load_signal_state(path)
    prev = state.get(symbol, {})
    prev_action = prev.get("action")
    regime = sig.get("action")
    payload = {
        "symbol": symbol,
        "verdict": sig.get("verdict"),
        "signal": regime or "none",
        "price": sig.get("price"),
        "score": sig.get("score"),
        "delta": sig.get("delta"),
        "candle_close": sig.get("candle_close"),
        "notified": False,
    }
    if not regime:
        payload["reason"] = "no_data"
        return payload
    if prev_action == regime:
        payload["reason"] = "unchanged"
        return payload
    # First observation of flat: nothing to clear, stay silent.
    if regime == "flat" and prev_action not in ("long", "short"):
        state[symbol] = {"action": "flat", "price": sig.get("price"), "at": now_str()}
        save_signal_state(state, path)
        payload["reason"] = "no_signal"
        return payload
    payload["notified"] = bool(send_fn(build_signal_embed(symbol, sig, previous=prev_action)))
    state[symbol] = {"action": regime, "price": sig.get("price"), "at": now_str()}
    save_signal_state(state, path)
    return payload


def now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
