import yaml
from pathlib import Path
from datetime import datetime, timezone

from fastapi import FastAPI
from data.fetcher import fetch_ohlcv
from indicators.calculator import compute_all
from analysis.summary import analyze_signals, generate_market_summary
from analysis.multi_tf import analyze_multi_timeframe
from monitor.state import load_state, save_state, RECORD_PATH
from monitor.checker import check_reversal, check_strong_signal
from monitor.notifier import (
    build_reversal_embed,
    build_strong_signal_embed,
    build_report_embed,
    send_webhook,
)

CONFIG_PATH = Path(__file__).parent / "config.yaml"

with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

app = FastAPI(title="Crypto Monitor")


def _today():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _process_symbol(symbol: str) -> dict:
    df = fetch_ohlcv(symbol, timeframe="1h", limit=200)
    result = compute_all(df)
    overlay = result["overlay"]
    subplots = result["subplots"]
    signals = analyze_signals(overlay, subplots)
    return {"overlay": overlay, "subplots": subplots, "signals": signals}


@app.get("/check")
def check():
    cfg = CONFIG["alerts"]
    webhook = CONFIG["discord"]["webhook_url"]
    state = load_state()
    today = _today()
    today_key = f"record_{today}"
    today_state = state.get(today_key, {})
    alerts_sent = []

    for symbol in CONFIG["symbols"]:
        try:
            result = _process_symbol(symbol)
        except Exception:
            continue

        new_sig = result["signals"]
        old_sym_state = today_state.get(symbol, {})

        # Level 1: trend reversal
        reversal = check_reversal(old_sym_state, new_sig, cfg["trend_reversal"])
        if reversal:
            changes = {"RSI": f"{result['subplots']['rsi']['RSI'].iloc[-1]:.1f}"}
            embed = build_reversal_embed(symbol, reversal["from"], reversal["to"], changes)
            try:
                send_webhook(webhook, embed)
                alerts_sent.append(f"{symbol}: reversal {reversal['from']}→{reversal['to']}")
            except Exception:
                pass

        # Level 2: strong signal
        strong = check_strong_signal(symbol, old_sym_state, new_sig, cfg["strong_signal_threshold"])
        if strong:
            embed = build_strong_signal_embed(symbol, strong["direction"],
                                              strong["bullish"], strong["total"])
            try:
                send_webhook(webhook, embed)
                alerts_sent.append(f"{symbol}: strong {strong['direction']}")
                new_sig["last_strong_notified"] = strong["direction"]
            except Exception:
                pass

        today_state[symbol] = new_sig

    state[today_key] = today_state
    save_state(state)
    return {"status": "ok", "alerts": alerts_sent, "time": datetime.now(timezone.utc).isoformat()}


@app.get("/report")
def report():
    cfg = CONFIG["alerts"]
    webhook = CONFIG["discord"]["webhook_url"]
    reports_sent = []

    for symbol in CONFIG["symbols"]:
        try:
            result = _process_symbol(symbol)
        except Exception:
            continue

        overlay = result["overlay"]
        subplots = result["subplots"]
        summary = generate_market_summary(overlay, subplots)
        tf_results = analyze_multi_timeframe(symbol)

        embed = build_report_embed(symbol, summary, tf_results)
        try:
            send_webhook(webhook, embed)
            reports_sent.append(symbol)
        except Exception:
            pass

    return {
        "status": "ok",
        "reports": reports_sent,
        "time": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/health")
def health():
    return {"status": "alive"}
