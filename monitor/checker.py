from typing import Optional

REVERSAL_TRIGGERS = {"偏多", "偏空"}


def check_reversal(
    old_state: dict,
    new_signals: dict,
    trend_reversal_enabled: bool,
) -> Optional[dict]:
    if not trend_reversal_enabled:
        return None
    old_dir = old_state.get("direction", "")
    new_dir = new_signals.get("direction", "")
    if old_dir in REVERSAL_TRIGGERS and new_dir in REVERSAL_TRIGGERS and old_dir != new_dir:
        return {"from": old_dir, "to": new_dir}
    return None


def check_strong_signal(
    symbol: str,
    old_state: dict,
    new_signals: dict,
    strong_signal_threshold: float,
) -> Optional[dict]:
    new_dir = new_signals.get("direction", "")
    if new_dir not in REVERSAL_TRIGGERS:
        return None
    bullish = new_signals.get("bullish_count", 0)
    bearish = new_signals.get("bearish_count", 0)
    total = new_signals.get("total") or bullish + bearish
    ratio = max(bullish, bearish) / total if total > 0 else 0
    if ratio >= strong_signal_threshold:
        last_notified = old_state.get("last_strong_notified")
        if last_notified != new_dir:
            return {"direction": new_dir, "bullish": bullish, "bearish": bearish, "total": total}
    return None
