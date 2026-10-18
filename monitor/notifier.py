import time
import os
import base64
import yaml
from pathlib import Path

import httpx
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

TAIWAN_TZ = ZoneInfo("Asia/Taipei")

CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

def _decode_b64(val: str) -> str:
    try:
        return base64.b64decode(val).decode()
    except Exception:
        return val

bot_token = os.environ.get("DISCORD_BOT_TOKEN", "")
channel_id = os.environ.get("DISCORD_CHANNEL_ID", "")
if not bot_token and CONFIG.get("discord", {}).get("bot_token_b64"):
    bot_token = _decode_b64(CONFIG["discord"]["bot_token_b64"])
if not channel_id and CONFIG.get("discord", {}).get("channel_id_b64"):
    channel_id = _decode_b64(CONFIG["discord"]["channel_id_b64"])
CONFIG["discord"]["bot_token"] = bot_token
CONFIG["discord"]["channel_id"] = channel_id

DISCORD_COLORS = {
    "reversal": 0xFFA500,
    "strong": 0x00FF00,
    "report": 0x4FC3F7,
}


def send_bot_message(embed: dict) -> bool:
    bot_token = CONFIG.get("discord", {}).get("bot_token", "")
    channel_id = CONFIG.get("discord", {}).get("channel_id", "")
    proxy = CONFIG.get("discord", {}).get("api_proxy", "").rstrip("/")
    if not bot_token or not channel_id:
        return False
    if proxy:
        url = f"{proxy}?path=/channels/{channel_id}/messages"
    else:
        url = f"https://discord.com/api/v10/channels/{channel_id}/messages"
    headers = {"Authorization": f"Bot {bot_token}"}
    try:
        resp = httpx.post(url, json=embed, headers=headers, timeout=15)
        if resp.status_code in (200, 201):
            return True
        print(f"[bot] API error: {resp.status_code} {resp.text[:200]}")
        return False
    except Exception as e:
        print(f"[bot] Send failed: {e}")
        return False


def build_reversal_embed(symbol: str, old_dir: str, new_dir: str, changes: dict) -> dict:
    fields = [{"name": k, "value": v, "inline": True} for k, v in changes.items()]
    return {"embeds": [{"title": f"🚨 {symbol} 趨勢反轉", "description": f"{old_dir} → **{new_dir}**", "color": DISCORD_COLORS["reversal"], "fields": fields}]}


def build_strong_signal_embed(symbol: str, direction: str, bullish: int, total: int) -> dict:
    emoji = "📈" if direction == "偏多" else "📉"
    return {"embeds": [{"title": f"{emoji} {symbol} {direction}訊號強烈", "description": f"多空比: {bullish}/{total} {direction}", "color": DISCORD_COLORS["strong"]}]}


def _interpret_delta(delta: float) -> str:
    if delta > 0.01:
        return "↑ 快速上升"
    elif delta > 0.001:
        return "↗ 穩定上升"
    elif delta > -0.001:
        return "→ 持平"
    elif delta > -0.01:
        return "↘ 穩定下降"
    else:
        return "↓ 快速下降"


def _report_body(symbol, summary, tf_results, momentum, momentum_scores) -> str:
    if isinstance(summary, dict):
        summary = "\n".join(f"{k}: {v}" for k, v in summary.items())
    lines = [f"**{symbol}**"]
    if summary:
        lines.append(str(summary))
    tf_lines = [f"{r['label']}: {r['direction']}" for r in (tf_results or [])]
    if tf_lines:
        lines.append("📈 多時間框架 " + " | ".join(tf_lines))
    if momentum:
        arrow = " → ".join(f"{s['direction']}({s['strength']})" for s in momentum["states"])
        lines.append(f"⚡ 動能演進: {arrow} — {momentum['label']}")
    if momentum_scores and len(momentum_scores) >= 3:
        prev3, prev2, prev1, current = momentum_scores[-4:]
        delta1, delta2, delta3 = current - prev1, prev1 - prev2, prev2 - prev3
        acceleration = delta1 - delta2
        trend_dir = "↑ 上升" if delta1 > 0.01 else "↓ 下降" if delta1 < -0.01 else "→ 持平"
        accel_label = "加速" if acceleration > 0.01 else "減速" if acceleration < -0.01 else "穩定"
        lines.append(f"📊 分數 {prev3:.2f} → {prev2:.2f} → {prev1:.2f} → {current:.2f}")
        lines.append(f"📐 微分 {delta3:+.3f} → {delta2:+.3f} → {delta1:+.3f}")
        lines.append(f"🎯 {trend_dir} | {accel_label} ({acceleration:+.3f})")
    elif momentum_scores:
        lines.append(f"📊 分數 {momentum_scores[-1]:.3f}")
    return "\n".join(lines)


def build_report_embed(symbol, summary, tf_results, momentum=None, momentum_scores=None, now=None) -> dict:
    triggered_at = (now or datetime.now()).astimezone(TAIWAN_TZ)
    body = _report_body(symbol, summary, tf_results, momentum, momentum_scores)
    return {"embeds": [{"title": f"📊 市場日報 — {symbol}", "description": f"🕐 {triggered_at:%Y/%m/%d %H:%M}\n{body}", "color": DISCORD_COLORS["report"]}]}


def build_combined_report_embed(sections, now=None) -> dict:
    """sections: [{symbol, summary, tf_results, momentum, momentum_scores}]. One Discord message."""
    triggered_at = (now or datetime.now()).astimezone(TAIWAN_TZ)
    names = " / ".join(s["symbol"].split("/")[0] for s in sections)
    parts = [f"🕐 {triggered_at:%Y/%m/%d %H:%M}"]
    for s in sections:
        parts.append(_report_body(s["symbol"], s.get("summary"), s.get("tf_results"), s.get("momentum"), s.get("momentum_scores")))
    description = "\n\n".join(parts)
    if len(description) > 4000:
        description = description[:3990] + "\n…"
    return {"embeds": [{"title": f"📊 市場日報 — {names}", "description": description, "color": DISCORD_COLORS["report"]}]}


def send_webhook(url: str, payload: dict, max_retries: int = 3) -> bool:
    if send_bot_message(payload):
        return True
    for attempt in range(max_retries):
        try:
            resp = httpx.post(url, json=payload, timeout=15)
            if resp.status_code == 429:
                retry_after = resp.json().get("retry_after", 5)
                time.sleep(max(retry_after + 2, 10 * (attempt + 1)))
                continue
            resp.raise_for_status()
            return True
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429 and attempt < max_retries - 1:
                retry_after = e.response.json().get("retry_after", 5)
                time.sleep(max(retry_after + 2, 10 * (attempt + 1)))
                continue
            raise
    return False
