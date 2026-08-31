import time

import httpx
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

TAIWAN_TZ = ZoneInfo("Asia/Taipei")

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


def build_report_embed(symbol: str, summary, tf_results: list,
                       momentum: Optional[dict] = None,
                       now: Optional[datetime] = None) -> dict:
    if isinstance(summary, dict):
        summary = "\n".join(f"{k}: {v}" for k, v in summary.items())
    tf_lines = [f"{r['label']}: {r['direction']}" for r in tf_results]
    triggered_at = (now or datetime.now()).astimezone(TAIWAN_TZ)
    description = f"🕐 {triggered_at:%Y/%m/%d %H:%M}"
    if summary:
        description += f"\n{summary}"
    if tf_lines:
        description += "\n\n📈 **多時間框架**\n" + " | ".join(tf_lines)
    if momentum:
        arrow = " → ".join(
            f"{s['direction']}({s['strength']})" for s in momentum["states"])
        description += f"\n\n⚡ **動能演進**: {arrow} — {momentum['label']}"
    return {
        "embeds": [{
            "title": f"📊 市場日報 — {symbol}",
            "description": description,
            "color": DISCORD_COLORS["report"],
        }]
    }


def send_webhook(url: str, payload: dict, max_retries: int = 3) -> bool:
    """Send webhook with retry logic for rate limits."""
    for attempt in range(max_retries):
        try:
            resp = httpx.post(url, json=payload, timeout=10)
            
            if resp.status_code == 429:
                retry_after = resp.json().get("retry_after", 1)
                print(f"Rate limited, waiting {retry_after}s...")
                time.sleep(retry_after + 1)
                continue
            
            resp.raise_for_status()
            return True
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429 and attempt < max_retries - 1:
                retry_after = e.response.json().get("retry_after", 1)
                print(f"Rate limited, waiting {retry_after}s...")
                time.sleep(retry_after + 1)
                continue
            raise
    
    return False
