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


def _interpret_delta(delta: float) -> str:
    """Interpret delta value for display."""
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


def build_report_embed(symbol: str, summary, tf_results: list,
                       momentum: Optional[dict] = None,
                       momentum_scores: Optional[list] = None,
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
    if momentum_scores and len(momentum_scores) >= 3:
        # momentum_scores: [score_3bars_ago, score_2bars_ago, score_1bar_ago, current_score]
        prev3, prev2, prev1, current = momentum_scores[-4:]
        
        # Calculate delta (first derivative) - trend direction
        delta1 = current - prev1
        delta2 = prev1 - prev2
        delta3 = prev2 - prev3
        
        # Calculate acceleration (second derivative) - trend strength
        acceleration = delta1 - delta2
        
        # Interpret deltas
        delta1_interp = _interpret_delta(delta1)
        delta2_interp = _interpret_delta(delta2)
        delta3_interp = _interpret_delta(delta3)
        
        # Determine trend direction
        if delta1 > 0.01:
            trend_dir = "↑ 上升"
        elif delta1 < -0.01:
            trend_dir = "↓ 下降"
        else:
            trend_dir = "→ 持平"
        
        # Determine acceleration
        if acceleration > 0.01:
            accel_label = "加速"
        elif acceleration < -0.01:
            accel_label = "減速"
        else:
            accel_label = "穩定"
        
        # Build score history
        score_history = f"{prev3:.2f} → {prev2:.2f} → {prev1:.2f} → {current:.2f}"
        
        # Build delta history with interpretation
        delta_history = f"{delta3:+.3f}({delta3_interp}) → {delta2:+.3f}({delta2_interp}) → {delta1:+.3f}({delta1_interp})"
        
        description += f"\n\n📊 **動能分數演進**: {score_history}"
        description += f"\n📐 **動能微分**: {delta_history}"
        description += f"\n🎯 **趨勢方向**: {trend_dir} | **加速度**: {accel_label} ({acceleration:+.3f})"
    elif momentum_scores and len(momentum_scores) > 0:
        current = momentum_scores[-1]
        score_pct = int((current + 1) * 50)
        bar = "█" * (score_pct // 5) + "░" * (20 - score_pct // 5)
        description += f"\n\n📊 **動能分數**: {current:.3f} [{bar}]"
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
            print(f"[webhook] attempt={attempt+1} url={url[:60]}...")
            resp = httpx.post(url, json=payload, timeout=10)
            print(f"[webhook] status={resp.status_code} body={resp.text[:200]}")
            
            if resp.status_code == 429:
                retry_after = resp.json().get("retry_after", 1)
                print(f"[webhook] Rate limited, waiting {retry_after}s...")
                time.sleep(retry_after + 1)
                continue
            
            resp.raise_for_status()
            print(f"[webhook] success")
            return True
            
        except httpx.HTTPStatusError as e:
            print(f"[webhook] HTTP error: {e.response.status_code} {e.response.text[:200]}")
            if e.response.status_code == 429 and attempt < max_retries - 1:
                retry_after = e.response.json().get("retry_after", 1)
                print(f"[webhook] Rate limited, waiting {retry_after}s...")
                time.sleep(retry_after + 1)
                continue
            raise
    
    return False
