from bot.state import init_state, load_state, save_state


def test_init_state_shape():
    s = init_state(["BTC/USDT:USDT", "SOL/USDT:USDT"], 10000.0)
    assert s["session_start_equity"] == 10000.0
    assert s["positions"]["BTC/USDT:USDT"] is None
    assert s["daily_loss_stopped"] is False
    assert s["realized_pnl"] == 0.0


def test_state_roundtrip(tmp_path):
    p = tmp_path / "state.json"
    s = init_state(["BTC/USDT:USDT"], 10000.0)
    s["positions"]["BTC/USDT:USDT"] = {"side": "long", "entry_price": 100.0,
                                        "qty": 50.0, "entry_time": "2026-09-06T10:00:00"}
    save_state(s, p)
    loaded = load_state(p)
    assert loaded["positions"]["BTC/USDT:USDT"]["side"] == "long"
    assert loaded["session_start_equity"] == 10000.0


def test_load_missing_returns_empty(tmp_path):
    assert load_state(tmp_path / "nope.json") == {}