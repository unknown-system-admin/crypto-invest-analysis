from monitor.notifier import build_reversal_embed, build_strong_signal_embed, build_report_embed, send_webhook


def test_build_reversal_embed_has_correct_fields():
    embed = build_reversal_embed("BTC/USDT", "偏多", "偏空", {"RSI": "72.3 → 45.1", "MACD": "多頭 → 空頭"})
    assert embed["embeds"][0]["title"] == "🚨 BTC/USDT 趨勢反轉"
    assert embed["embeds"][0]["color"] == 0xFFA500


def test_build_strong_signal_embed_contains_direction():
    embed = build_strong_signal_embed("SOL/USDT", "偏多", 4, 5)
    assert embed["embeds"][0]["title"] == "📈 SOL/USDT 偏多訊號強烈"
    desc = embed["embeds"][0]["description"]
    assert "4/5 偏多" in desc


def test_report_embed_includes_summary_and_timeframes():
    embed = build_report_embed("ETH/USDT", "整體偏多", [
        {"label": "15m", "direction": "偏多"},
        {"label": "1h", "direction": "偏空"},
    ])
    assert embed["embeds"][0]["title"] == "📊 市場日報 — ETH/USDT"
    desc = embed["embeds"][0]["description"]
    assert "整體偏多" in desc
    assert "15m" in desc
    assert "1h" in desc


def test_report_embed_includes_momentum():
    momentum = {
        "label": "減弱中",
        "states": [
            {"direction": "偏多", "strength": "強"},
            {"direction": "偏多", "strength": "中"},
            {"direction": "偏多", "strength": "弱"},
        ],
    }
    embed = build_report_embed("BTC/USDT", "整體偏多", [], momentum=momentum)
    desc = embed["embeds"][0]["description"]
    assert "動能演進" in desc
    assert "減弱中" in desc
