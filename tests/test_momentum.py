from analysis.momentum import classify_state, momentum_trend


def test_classify_strong_bullish():
    signals = {"direction": "偏多", "bullish_count": 4, "bearish_count": 0, "total": 4}
    s = classify_state(signals)
    assert s["direction"] == "偏多"
    assert s["strength"] == "強"


def test_classify_medium_bullish():
    signals = {"direction": "偏多", "bullish_count": 2, "bearish_count": 1, "total": 3}
    assert classify_state(signals)["strength"] == "中"


def test_classify_weak_bullish():
    signals = {"direction": "偏多", "bullish_count": 1, "bearish_count": 2, "total": 3}
    assert classify_state(signals)["strength"] == "弱"


def test_classify_boundary_075_is_strong():
    signals = {"direction": "偏多", "bullish_count": 3, "bearish_count": 1, "total": 4}
    assert classify_state(signals)["strength"] == "強"


def test_classify_boundary_05_is_medium():
    signals = {"direction": "偏空", "bullish_count": 3, "bearish_count": 3, "total": 6}
    s = classify_state(signals)
    assert s["strength"] == "中"
    assert s["direction"] == "偏空"


def test_trend_weakening():
    states = [
        {"direction": "偏多", "strength": "強"},
        {"direction": "偏多", "strength": "中"},
        {"direction": "偏多", "strength": "弱"},
    ]
    trend = momentum_trend(states)
    assert trend["label"] == "減弱中"
    assert trend["trend"] == "weakening"


def test_trend_strengthening():
    states = [
        {"direction": "偏空", "strength": "弱"},
        {"direction": "偏空", "strength": "中"},
        {"direction": "偏空", "strength": "強"},
    ]
    assert momentum_trend(states)["label"] == "增強中"
    assert momentum_trend(states)["trend"] == "strengthening"


def test_trend_stable():
    states = [
        {"direction": "偏多", "strength": "強"},
        {"direction": "偏多", "strength": "強"},
        {"direction": "偏多", "strength": "強"},
    ]
    trend = momentum_trend(states)
    assert trend["label"] == "維持"
    assert trend["trend"] == "stable"


def test_trend_reversal():
    states = [
        {"direction": "偏空", "strength": "強"},
        {"direction": "偏多", "strength": "中"},
        {"direction": "偏多", "strength": "強"},
    ]
    assert momentum_trend(states)["label"] == "方向反轉"
    assert momentum_trend(states)["trend"] == "reversal"