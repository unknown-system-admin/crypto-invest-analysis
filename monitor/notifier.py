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
    """Send message using Discord Bot REST API (no gateway needed)."""
    bot_token = CONFIG.get("discord", {}).get("bot_token", "")
    channel_id = CONFIG.get("discord", {}).get("channel_id", "")

    if not bot_token or not channel_id:
        return False

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


def build_report_embed(symbol: str, summary: str, tf_results: list,
                       momentum: Optional[str] = None,
                       momentum_scores: Optional[list] = None) -> dict:
    now = datetime.now(TAIWAN_TZ)
    description = f"**{summary}**\n\n🕐 {now.strftime('%Y/%m/%d %H:%M')}"

    if tf_results:
        description += "\n\n**跨級別分析：**"
        for r in tf_results:
            if r.get("error"):
                description += f"\n• {r['label']}: ❌"
            else:
                description += f"\n• {r['label']}: {r['direction']} (RSI {r['rsi']:.1f})"

    if momentum:
        description += f"\n\n**動能趨勢：** {momentum}"

    if momentum_scores and len(momentum_scores) >= 2:
        prev = momentum_scores[-2]
        curr = momentum_scores[-1]
        delta = curr - prev
        if len(momentum_scores) >= 3:
            prev2 = momentum_scores[-3]
            accel = delta - (prev - prev2)
        else:
            accel = 0

        if delta > 0.01:
            delta_label = "↑快速上升"
        elif delta > 0.001:
            delta_label = "↗穩定上升"
        elif delta > -0.001:
            delta_label = "→持平"
        elif delta > -0.01:
            delta_label = "↘穩定下降"
        else:
            delta_label = "↓快速下降"

        description += f"\n📊 **動能分數演化：** {prev:.3f} → {curr:.3f}"
        description += f"\n📈 **1階導數（變化率）：** {delta:+.4f} ({delta_label})"
        description += f"\n📉 **2階導數（加速度）：** {accel:+.5f}"

    return {
        "embeds": [{
            "title": f"📊 市場日報 — {symbol}",
            "description": description,
            "color": DISCORD_COLORS["report"],
        }]
    }


def send_webhook(url: str, payload: dict, max_retries: int = 3) -> bool:
    """Send webhook with bot fallback."""
    bot_ok = send_bot_message(payload)
    if bot_ok:
        return True

    for attempt in range(max_retries):
        try:
            resp = httpx.post(url, json=payload, timeout=15)
            if resp.status_code == 429:
                retry_after = resp.json().get("retry_after", 5)
                wait = max(retry_after + 2, 10 * (attempt + 1))
                print(f"[webhook] 429 rate limited, waiting {wait}s (attempt {attempt+1})")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return True
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429 and attempt < max_retries - 1:
                retry_after = e.response.json().get("retry_after", 5)
                wait = max(retry_after + 2, 10 * (attempt + 1))
                print(f"[webhook] 429 rate limited, waiting {wait}s (attempt {attempt+1})")
                time.sleep(wait)
            else:
                print(f"[webhook] Error: {e}")
                return False
        except Exception as e:
            print(f"[webhook] Error: {e}")
            return False
    return False
