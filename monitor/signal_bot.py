"""Live signal notifier.

Price is the exchange last trade. No virtual position.
BTC and SOL are sent in one Discord message when any regime changes.
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

LABELS = {"long": "做多", "short": "做空", "flat": "趨勢消失"}


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


def _line(symbol: str, sig: dict, previous: str = None, changed: bool = False) -> str:
    label = LABELS.get(sig.get("action"), sig.get("action"))
    mark = "◆ " if changed else ""
    text = (
        f"{mark}**{symbol}** {label} · ${sig['price']:,.2f}\n"
        f"分數 {sig['score']:.3f} · 變化 {sig.get('delta', 0):+.3f}"
    )
    if changed and previous and previous != sig.get("action"):
        text += f"\n{LABELS.get(previous, previous)} → {label}"
    return text


def build_basket_embed(rows: list) -> dict:
    changed = [r for r in rows if r.get("changed")]
    title = "策略訊號 · " + " / ".join(r["symbol"].split("/")[0] for r in rows)
    lines = [_line(r["symbol"], r["sig"], r.get("previous"), r.get("changed")) for r in rows]
    lines.append("\n門檻：做多 > 0.05 且變化>0 · 做空 < -0.30 且變化<0")
    lines.append("◆ 為本次變更。沒標記的是同次狀態，供對照。")
    lines.append("\n👉 請自行至 OKX 下單（通知，非自動執行）")
    color = 0x00FF00 if any(r["sig"].get("action") == "long" for r in changed) else 0xFF0000 if any(r["sig"].get("action") == "short" for r in changed) else 0xAAAAAA
    return {"embeds": [{"title": title, "description": "\n".join(lines), "color": color}]}


def evaluate_basket(items, send_fn, path=SIGNAL_STATE_PATH, cfg: dict = None) -> dict:
    """items: [{symbol, features}]. One Discord message if any regime changes."""
    cfg = cfg or DEFAULT_CFG
    state = load_signal_state(path)
    results = []
    rows = []
    for item in items:
        symbol = item["symbol"]
        try:
            sig = compute_latest_signal(item["features"], symbol, buy=cfg["buy"], sell=cfg["sell"])
        except Exception as e:
            results.append({"symbol": symbol, "error": str(e), "notified": False})
            continue
        prev = state.get(symbol, {}).get("action")
        regime = sig.get("action")
        changed = bool(regime) and prev != regime and not (regime == "flat" and prev not in ("long", "short"))
        results.append({
            "symbol": symbol,
            "verdict": sig.get("verdict"),
            "signal": regime or "none",
            "price": sig.get("price"),
            "score": sig.get("score"),
            "delta": sig.get("delta"),
            "candle_close": sig.get("candle_close"),
            "changed": changed,
            "notified": False,
        })
        if regime:
            rows.append({"symbol": symbol, "sig": sig, "previous": prev, "changed": changed})
            state[symbol] = {"action": regime, "price": sig.get("price"), "at": now_str()}
    if any(r["changed"] for r in rows):
        ok = bool(send_fn(build_basket_embed(rows)))
        for r in results:
            if r.get("changed"):
                r["notified"] = ok
        save_signal_state(state, path)
    else:
        save_signal_state(state, path)
    return {"results": results, "notified": [r["symbol"] for r in results if r.get("notified")]}


def now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
