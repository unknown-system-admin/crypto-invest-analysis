import json
from monitor.state import load_state, save_state


def test_state_roundtrip(tmp_path):
    path = tmp_path / "record.json"
    data = {"BTC/USDT": {"direction": "偏多", "bullish_count": 4}}
    save_state(data, path)
    loaded = load_state(path)
    assert loaded == data


def test_load_missing_file_returns_empty(tmp_path):
    path = tmp_path / "nonexistent.json"
    result = load_state(path)
    assert result == {}
