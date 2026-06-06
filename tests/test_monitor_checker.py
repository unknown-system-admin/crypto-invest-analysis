from monitor.checker import check_reversal, check_strong_signal


def test_check_reversal_detects_change():
    old = {"direction": "偏多", "bullish_count": 4, "bearish_count": 1}
    new = {"direction": "偏空", "bullish_count": 1, "bearish_count": 4}
    result = check_reversal(old, new, trend_reversal_enabled=True)
    assert result is not None
    assert "偏多" in result["from"]
    assert "偏空" in result["to"]


def test_check_reversal_no_change():
    old = {"direction": "偏多", "bullish_count": 4, "bearish_count": 1}
    new = {"direction": "偏多", "bullish_count": 3, "bearish_count": 2}
    result = check_reversal(old, new, trend_reversal_enabled=True)
    assert result is None


def test_check_reversal_disabled():
    old = {"direction": "偏多", "bullish_count": 4, "bearish_count": 1}
    new = {"direction": "偏空", "bullish_count": 1, "bearish_count": 4}
    result = check_reversal(old, new, trend_reversal_enabled=False)
    assert result is None


def test_check_strong_signal_triggers():
    old = {"direction": "偏多", "bullish_count": 3, "bearish_count": 2}
    new = {"direction": "偏多", "bullish_count": 4, "bearish_count": 1}
    result = check_strong_signal("BTC/USDT", old, new, strong_signal_threshold=0.75)
    assert result is not None
    assert result["direction"] == "偏多"


def test_check_strong_signal_below_threshold():
    old = {"direction": "偏多", "bullish_count": 3, "bearish_count": 2}
    new = {"direction": "偏多", "bullish_count": 3, "bearish_count": 2}
    result = check_strong_signal("BTC/USDT", old, new, strong_signal_threshold=0.75)
    assert result is None


def test_check_strong_signal_already_notified():
    old = {"direction": "偏多", "bullish_count": 4, "bearish_count": 1,
           "last_strong_notified": "偏多"}
    new = {"direction": "偏多", "bullish_count": 4, "bearish_count": 1}
    result = check_strong_signal("BTC/USDT", old, new, strong_signal_threshold=0.75)
    assert result is None
