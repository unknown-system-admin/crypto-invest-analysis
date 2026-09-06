import time
from datetime import datetime, timezone

from bot.signals import evaluate
from bot.state import STATE_PATH, init_state, load_state, save_state
from bot.risk import (check_daily_loss, check_stop_loss,
                      check_position_limit, position_notional)


def _equity_estimate(state):
    """Approximate equity: start + realized + sum of unrealized across all
    open positions (each position stores its own mark-to-market)."""
    unrealized = 0.0
    for pos in state["positions"].values():
        if pos is None:
            continue
        cur = pos.get("unrealized")
        if cur is None:
            cur = 0.0
        unrealized += cur
    return state["session_start_equity"] + state["realized_pnl"] + unrealized


def _mark_unrealized(pos, price):
    """Store the position's mark-to-market at the given price."""
    if not pos.get("entry_price"):
        return
    if pos["side"] == "long":
        pos["unrealized"] = (price - pos["entry_price"]) * pos["qty"]
    else:
        pos["unrealized"] = (pos["entry_price"] - price) * pos["qty"]


def run_bot(cfg, executor, fetch_fn, state_path=None,
            max_iterations=None, logger=print):
    if state_path is None:
        state_path = STATE_PATH
    state = load_state(state_path) or init_state(cfg.symbols, cfg.initial_capital)
    if state.get("started_at") is None:
        state["started_at"] = datetime.now(timezone.utc).isoformat()

    # --- new UTC day: begin a fresh session, preserving positions ---
    try:
        started = datetime.fromisoformat(state["started_at"])
    except (TypeError, ValueError):
        started = None
    if started is not None:
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        if started.date() != datetime.now(timezone.utc).date():
            state["session_start_equity"] = state["session_start_equity"] + state["realized_pnl"]
            state["realized_pnl"] = 0.0
            state["daily_loss_stopped"] = False
            state["started_at"] = datetime.now(timezone.utc).isoformat()
            logger("New session day detected; reset daily loss stop")

    iteration = 0
    try:
        while max_iterations is None or iteration < max_iterations:
            iteration += 1
            for symbol in cfg.symbols:
                try:
                    _tick(cfg, executor, fetch_fn, state, symbol, logger)
                except Exception as e:
                    logger(f"[{symbol}] ERROR: {e}")
            save_state(state, state_path)
            if max_iterations is None:
                time.sleep(cfg.poll_seconds)
    except KeyboardInterrupt:
        logger("Stopping (Ctrl+C). Positions preserved for next launch.")
    save_state(state, state_path)
    return state


def _tick(cfg, executor, fetch_fn, state, symbol, logger):
    df_htf = fetch_fn(symbol, cfg.htf_timeframe, cfg.htf_candles)
    df_ltf = fetch_fn(symbol, cfg.ltf_timeframe, cfg.ltf_candles)
    price = float(df_ltf["close"].iloc[-1])

    pos = state["positions"][symbol]
    if pos is not None:
        _mark_unrealized(pos, price)
    side = pos["side"] if pos else "flat"

    # --- daily loss kill switch ---
    equity = _equity_estimate(state)
    daily = check_daily_loss(state["session_start_equity"], equity,
                             cfg.max_daily_loss_pct)
    if not daily.allowed:
        if not state["daily_loss_stopped"]:
            logger(f"[{symbol}] {daily.reason} -> flattening & stopping for the day")
            state["daily_loss_stopped"] = True
            if pos and not cfg.dry_run:
                executor.close_position(symbol, pos["side"])
            state["positions"][symbol] = None
        return

    # --- per-trade stop loss ---
    if pos is not None:
        if check_stop_loss(pos["entry_price"], price, pos["side"], cfg.stop_loss_pct):
            logger(f"[{symbol}] stop loss hit ({cfg.stop_loss_pct:.1%}) @ {price:.2f}")
            _realize(state, symbol, pos, price)
            if not cfg.dry_run:
                executor.close_position(symbol, pos["side"])
            return

    # --- position limit (entry-only) ---
    # When a position is already open the symbol is counted in open_symbols and
    # would block the gate, so it must not run: exits still need to be evaluated.
    if pos is None:
        open_symbols = [s for s, p in state["positions"].items() if p is not None]
        limit = check_position_limit(open_symbols, symbol)
        if not limit.allowed:
            return

    # --- signal ---
    sig = evaluate(df_htf, df_ltf, cfg.htf_threshold, side)
    if sig.action == "none":
        return

    if sig.action in ("enter_long", "enter_short"):
        notional = position_notional(equity, cfg.margin_pct, cfg.leverage)
        qty = notional / price
        logger(f"[{symbol}] {sig.action} @ {price:.2f} (score={sig.htf_score:.3f})")
        if cfg.dry_run:
            state["positions"][symbol] = {"side": "long" if sig.action == "enter_long" else "short",
                                          "entry_price": price, "qty": qty,
                                          "entry_time": datetime.now(timezone.utc).isoformat()}
        else:
            executor.set_leverage(symbol, cfg.leverage)
            if sig.action == "enter_long":
                executor.open_long(symbol, notional)
            else:
                executor.open_short(symbol, notional)
            state["positions"][symbol] = {"side": "long" if sig.action == "enter_long" else "short",
                                          "entry_price": price, "qty": qty,
                                          "entry_time": datetime.now(timezone.utc).isoformat()}

    elif sig.action in ("exit_long", "exit_short"):
        logger(f"[{symbol}] {sig.action} @ {price:.2f}")
        _realize(state, symbol, pos, price)
        if not cfg.dry_run:
            executor.close_position(symbol, pos["side"])


def _realize(state, symbol, pos, price):
    if pos is None:
        return
    if pos["side"] == "long":
        pnl = (price - pos["entry_price"]) * pos["qty"]
    else:
        pnl = (pos["entry_price"] - price) * pos["qty"]
    state["realized_pnl"] += pnl
    state["positions"][symbol] = None