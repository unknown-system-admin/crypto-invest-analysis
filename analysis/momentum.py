STRENGTH_STRONG = 0.75
STRENGTH_MEDIUM = 0.5


def classify_state(signals: dict) -> dict:
    direction = signals["direction"]
    ratio = signals["bullish_count"] / signals["total"]
    if ratio >= STRENGTH_STRONG:
        strength = "強"
    elif ratio >= STRENGTH_MEDIUM:
        strength = "中"
    else:
        strength = "弱"
    return {"direction": direction, "strength": strength}


def _strength_key(strength: str) -> int:
    return {"強": 2, "中": 1, "弱": 0}.get(strength, 1)


def momentum_trend(states: list) -> dict:
    """states: oldest → newest，回傳 {label, trend, states}"""
    dirs = [s["direction"] for s in states]
    if len(set(dirs)) > 1:
        return {"label": "方向反轉", "trend": "reversal", "states": states}
    scores = [_strength_key(s["strength"]) for s in states]
    if scores == sorted(scores) and len(set(scores)) > 1:
        return {"label": "增強中", "trend": "strengthening", "states": states}
    if scores == sorted(scores, reverse=True) and len(set(scores)) > 1:
        return {"label": "減弱中", "trend": "weakening", "states": states}
    return {"label": "維持", "trend": "stable", "states": states}