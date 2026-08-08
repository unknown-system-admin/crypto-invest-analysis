import pandas as pd

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
    strict_up = all(a < b for a, b in zip(scores, scores[1:]))
    strict_down = all(a > b for a, b in zip(scores, scores[1:]))
    if strict_up:
        return {"label": "增強中", "trend": "strengthening", "states": states}
    if strict_down:
        return {"label": "減弱中", "trend": "weakening", "states": states}
    return {"label": "維持", "trend": "stable", "states": states}


def states_from_df(df: pd.DataFrame, step: int) -> list:
    """回傳 oldest → newest 的三個 momentum state"""
    from indicators.calculator import compute_all
    from analysis.summary import analyze_signals
    offsets = [0, step, 2 * step]
    states = []
    for off in offsets:
        df_view = df.iloc[: len(df) - off] if off > 0 else df
        if len(df_view) < 30:
            states.append({"direction": "震盪", "strength": "弱"})
            continue
        res = compute_all(df_view)
        sig = analyze_signals(res["overlay"], res["subplots"])
        states.append(classify_state(sig))
    return list(reversed(states))