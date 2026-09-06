from dataclasses import dataclass


@dataclass
class RiskDecision:
    allowed: bool
    reason: str = ""


def check_daily_loss(session_start_equity: float, current_equity: float,
                     max_daily_loss_pct: float) -> RiskDecision:
    if session_start_equity <= 0:
        return RiskDecision(False, "invalid start equity")
    loss = (session_start_equity - current_equity) / session_start_equity
    if loss >= max_daily_loss_pct:
        return RiskDecision(False, f"daily loss {loss:.1%} >= {max_daily_loss_pct:.0%}")
    return RiskDecision(True)


def check_stop_loss(entry_price: float, current_price: float,
                    side: str, stop_loss_pct: float) -> bool:
    if entry_price <= 0:
        return False
    if side == "long":
        return (entry_price - current_price) / entry_price >= stop_loss_pct
    return (current_price - entry_price) / entry_price >= stop_loss_pct


def position_notional(equity: float, margin_pct: float, leverage: int) -> float:
    return equity * margin_pct * leverage


def check_position_limit(open_positions: list, symbol: str,
                         max_per_symbol: int = 1) -> RiskDecision:
    count = sum(1 for s in open_positions if s == symbol)
    if count >= max_per_symbol:
        return RiskDecision(False, f"{symbol} already has {count} position(s)")
    return RiskDecision(True)


def trailing_stop_price(entry_price: float, peak_price: float, atr: float,
                        side: str, k: float, hard_stop_pct: float) -> float:
    if side == "long":
        atr_stop = peak_price - k * atr
        hard_stop = entry_price * (1 - hard_stop_pct)
        return max(atr_stop, hard_stop)
    atr_stop = peak_price + k * atr
    hard_stop = entry_price * (1 + hard_stop_pct)
    return min(atr_stop, hard_stop)


def check_trailing_stop(entry_price: float, peak_price: float, current_price: float,
                        atr: float, side: str, k: float, hard_stop_pct: float) -> bool:
    stop = trailing_stop_price(entry_price, peak_price, atr, side, k, hard_stop_pct)
    if side == "long":
        return current_price <= stop
    return current_price >= stop