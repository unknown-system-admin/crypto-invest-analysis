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