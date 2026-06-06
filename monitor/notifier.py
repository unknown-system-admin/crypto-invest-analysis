import httpx

DISCORD_COLORS = {
    "reversal": 0xFFA500,
    "strong": 0x00FF00,
    "report": 0x4FC3F7,
}


def build_reversal_embed(symbol: str, old_dir: str, new_dir: str, changes: dict) -> dict:
    fields = [{"name": k, "value": v, "inline": True} for k, v in changes.items()]
    return {
        "embeds": [{
            "title": f"🚨 {symbol} 趨勢反轉",
            "description": f"{old_dir} → **{new_dir}**",
            "color": DISCORD_COLORS["reversal"],
            "fields": fields,
        }]
    }


def build_strong_signal_embed(symbol: str, direction: str, bullish: int, total: int) -> dict:
    emoji = "📈" if direction == "偏多" else "📉"
    return {
        "embeds": [{
            "title": f"{emoji} {symbol} {direction}訊號強烈",
            "description": f"多空比: {bullish}/{total} {direction}",
            "color": DISCORD_COLORS["strong"],
        }]
    }


def build_report_embed(symbol: str, summary: str, tf_results: list) -> dict:
    tf_lines = [f"{r['label']}: {r['direction']}" for r in tf_results]
    description = summary + "\n\n📈 **多時間框架**\n" + " | ".join(tf_lines)
    return {
        "embeds": [{
            "title": f"📊 市場日報 — {symbol}",
            "description": description,
            "color": DISCORD_COLORS["report"],
        }]
    }


def send_webhook(url: str, payload: dict) -> bool:
    resp = httpx.post(url, json=payload, timeout=10)
    resp.raise_for_status()
    return True
