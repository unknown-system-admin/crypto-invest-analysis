from analysis.momentum import classify_state


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