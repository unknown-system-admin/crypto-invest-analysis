import yaml
import os
import base64
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI
from data.fetcher import fetch_ohlcv
from indicators.calculator import compute_all
from analysis.summary import analyze_signals, generate_market_summary
from analysis.multi_tf import analyze_multi_timeframe
from analysis.momentum import momentum_trend, states_from_df
from trading.strategy import CustomComposite
from trading.portfolio import PortfolioStore, calculate_position_size
from trading.executor import PaperExecutor, LiveExecutor
from trading.config import DEFAULT_RISK
from monitor.state import load_state, save_state, RECORD_PATH
from monitor.checker import check_reversal, check_strong_signal
from monitor.notifier import (
    build_reversal_embed,
    build_strong_signal_embed,
    build_report_embed,
    send_webhook,
    send_bot_message,
)

CONFIG_PATH = Path(__file__).parent / "config.yaml"

with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

# Decode base64 bot token and channel ID
def _decode_b64(val: str) -> str:
    try:
        return base64.b64decode(val).decode()
    except Exception:
        return val

bot_token = os.environ.get("DISCORD_BOT_TOKEN", "")
channel_id = os.environ.get("DISCORD_CHANNEL_ID", "")
if not bot_token and CONFIG.get("discord", {}).get("bot_token_b64"):
    bot_token = _decode_b64(CONFIG["discord"]["bot_token_b64"])
if not channel_id and CONFIG.get("discord", {}).get("channel_id_b64"):
    channel_id = _decode_b64(CONFIG["discord"]["channel_id_b64"])
CONFIG["discord"]["bot_token"] = bot_token
CONFIG["discord"]["channel_id"] = channel_id

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
    errors = []

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
        try:
            strong = check_strong_signal(symbol, old_sym_state, new_sig, cfg["strong_signal_threshold"])
        except Exception as e:
            errors.append(f"{symbol}: check_strong_signal failed: {e}")
            strong = None
        if strong:
            embed = build_strong_signal_embed(symbol, strong["direction"],
                                              strong["bullish"], strong["total"])
            try:
                send_webhook(webhook, embed)
                alerts_sent.append(f"{symbol}: strong {strong['direction']}")
                new_sig["last_strong_notified"] = strong["direction"]
            except Exception:
                pass

        # Level 3: trading strategy evaluation
        trading_cfg = CONFIG.get("trading", {})
        if trading_cfg.get("strategies"):
            mode = trading_cfg.get("mode", "paper")
            risk = trading_cfg.get("risk", DEFAULT_RISK)
            threshold = risk.get("signal_threshold", 0.6)

            strategy = CustomComposite(trading_cfg["strategies"], threshold=threshold)
            store = PortfolioStore()
            portfolio = store.load()

            signal = strategy.evaluate(result["overlay"], result["subplots"])
            if signal and signal.direction != "中立":
                current_price = float(result["overlay"]["close"].iloc[-1])
                has_pos = any(p.symbol == symbol for p in portfolio.positions)

                if mode == "live":
                    import ccxt
                    exchange = ccxt.binance()
                    executor = LiveExecutor(portfolio, exchange=exchange)
                else:
                    executor = PaperExecutor(portfolio)

                today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                daily_count = sum(1 for o in portfolio.orders if o.timestamp.startswith(today_str))

                if signal.direction == "偏多" and not has_pos and executor.can_trade(symbol, daily_count, risk.get("max_daily_trades", 10)):
                    qty = calculate_position_size(portfolio.cash, current_price, risk.get("max_position_pct", 25))
                    if qty > 0:
                        order = executor.execute_buy(symbol, current_price, qty)
                        embed = {
                            "embeds": [{
                                "title": f"🟢 {symbol} 買入訊號",
                                "description": f"價格: ${current_price:,.2f}\n數量: {qty}\n信心: {signal.confidence:.0%}",
                                "color": 0x00FF00,
                            }]
                        }
                        try:
                            send_webhook(webhook, embed)
                            alerts_sent.append(f"{symbol}: BUY signal")
                        except Exception:
                            pass

                elif signal.direction == "偏空" and has_pos and executor.can_trade(symbol, daily_count, risk.get("max_daily_trades", 10)):
                    pos_list = [p for p in portfolio.positions if p.symbol == symbol]
                    if pos_list:
                        pos = pos_list[0]
                        order = executor.execute_sell(symbol, current_price, pos.quantity)
                        if order:
                            embed = {
                                "embeds": [{
                                    "title": f"🔴 {symbol} 賣出訊號",
                                    "description": f"價格: ${current_price:,.2f}\n數量: {pos.quantity}\n損益: ${order.pnl:+.2f} ({order.pnl_pct:+.2f}%)",
                                    "color": 0xFF0000,
                                }]
                            }
                            try:
                                send_webhook(webhook, embed)
                                alerts_sent.append(f"{symbol}: SELL signal (PnL: ${order.pnl:+.2f})")
                            except Exception:
                                pass

                for p in portfolio.positions:
                    p.update_market(current_price)

                store.save(portfolio)

        today_state[symbol] = new_sig

    state[today_key] = today_state
    save_state(state)
    return {"status": "ok", "alerts": alerts_sent, "time": datetime.now(timezone.utc).isoformat()}


@app.get("/check_debug")
def check_debug():
    errors = {}
    for symbol in CONFIG["symbols"]:
        try:
            result = _process_symbol(symbol)
            errors[symbol] = "fetch_ohlcv + compute_all OK"
        except Exception as e:
            errors[symbol] = f"fetch_ohlcv failed: {e}"
            continue
        try:
            from analysis.summary import generate_market_summary
            summary = generate_market_summary(result["overlay"], result["subplots"])
            errors[symbol] = f"generate_market_summary OK (type={type(summary).__name__}, len={len(summary)})"
        except Exception as e:
            errors[symbol] = f"generate_market_summary failed: {e}"
    return {"errors": errors, "time": datetime.now(timezone.utc).isoformat()}


@app.get("/report")
def report(tf: Optional[str] = None, step: Optional[int] = None):
    rcfg = CONFIG.get("reports", {})
    tf = tf or rcfg.get("default_tf", "1d")
    step = step or rcfg.get("default_step", 1)
    webhook = CONFIG["discord"]["webhook_url"]
    reports_sent = []
    errors = []

    for symbol in CONFIG["symbols"]:
        try:
            df = fetch_ohlcv(symbol, timeframe=tf, limit=220)
        except Exception as e:
            errors.append(f"{symbol}: fetch/compute failed: {e}")
            continue
        try:
            comp = compute_all(df)
            summary = generate_market_summary(comp["overlay"], comp["subplots"])
        except Exception as e:
            errors.append(f"{symbol}: generate_market_summary failed: {e}")
            continue
        try:
            tf_results = analyze_multi_timeframe(symbol)
        except Exception:
            tf_results = []
        try:
            states = states_from_df(df, step)
            mtrend = momentum_trend(states)
        except Exception as e:
            errors.append(f"{symbol}: momentum analysis failed: {e}")
            mtrend = None
        try:
            from feature_engine.momentum import momentum_score
            from feature_engine.indicators import compute_all_indicators
            df_ind = compute_all_indicators(df)
            df_ind['close'] = df['close']  # Add close column for momentum_score
            scores = momentum_score(df_ind)
            # Get last 4 scores for evolution display
            recent_scores = scores.dropna().tolist()[-4:] if len(scores.dropna()) >= 4 else scores.dropna().tolist()
        except Exception as e:
            recent_scores = None

        try:
            embed = build_report_embed(symbol, summary, tf_results, 
                                      momentum=mtrend, 
                                      momentum_scores=recent_scores)
            ok = send_bot_message(embed)
            if ok:
                reports_sent.append(symbol)
            else:
                errors.append(f"{symbol}: send_bot_message failed")
        except Exception as e:
            errors.append(f"{symbol}: send failed: {e}")
    return {"status": "ok",
            "reports": reports_sent, "errors": errors,
            "time": datetime.now(timezone.utc).isoformat()}


@app.get("/health")
def health():
    return {"status": "alive"}


@app.get("/debug/webhook")
def debug_webhook():
    webhook = CONFIG["discord"]["webhook_url"]
    import httpx
    try:
        resp = httpx.get(webhook, timeout=10)
        return {
            "webhook_url_prefix": webhook[:80] + "...",
            "discord_status": resp.status_code,
            "discord_body": resp.text[:500],
        }
    except Exception as e:
        return {
            "webhook_url_prefix": webhook[:80] + "...",
            "error": str(e),
        }


@app.get("/debug/bot")
def debug_bot():
    import base64
    all_keys = list(os.environ.keys())
    
    # Show decoded config
    bot_token = CONFIG.get("discord", {}).get("bot_token", "")
    channel_id = CONFIG.get("discord", {}).get("channel_id", "")
    
    return {
        "total_env_vars": len(all_keys),
        "bot_token_prefix": bot_token[:20] + "..." if bot_token else "EMPTY",
        "channel_id": channel_id if channel_id else "EMPTY",
        "config_source": "env" if os.environ.get("DISCORD_BOT_TOKEN") else "b64_decode",
    }
