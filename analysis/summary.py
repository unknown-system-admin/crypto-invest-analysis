import pandas as pd
import numpy as np
from typing import Optional


def _ma_trend(overlay: pd.DataFrame) -> str:
    close = overlay["close"].iloc[-1]
    sma20 = overlay["SMA_20"].iloc[-1]
    sma50 = overlay["SMA_50"].iloc[-1]
    sma200 = overlay["SMA_200"].iloc[-1]

    parts = []
    if close > sma20 > sma50 > sma200:
        parts.append("多頭排列（SMA 20 > 50 > 200），趨勢強勁")
    elif close < sma20 < sma50 < sma200:
        parts.append("空頭排列（SMA 20 < 50 < 200），趨勢疲弱")
    elif close > sma20 > sma50:
        parts.append("短中期均線向上，動能偏多")
    elif close < sma20 < sma50:
        parts.append("短中期均線向下，動能偏空")
    else:
        parts.append("均線交錯，趨勢不明朗")

    ema12 = overlay["EMA_12"].iloc[-1]
    ema26 = overlay["EMA_26"].iloc[-1]
    if ema12 > ema26:
        parts.append("EMA12 在 EMA26 上方，短期偏多")
    else:
        parts.append("EMA12 在 EMA26 下方，短期偏空")

    return "；".join(parts)


def _rsi_analysis(subplots: dict) -> str:
    rsi = subplots["rsi"]["RSI"].iloc[-1]
    if pd.isna(rsi):
        return "RSI 數據不足"
    if rsi > 70:
        return f"RSI = {rsi:.1f}（超買區 > 70），可能面臨回調壓力"
    elif rsi < 30:
        return f"RSI = {rsi:.1f}（超賣區 < 30），可能出現反彈機會"
    elif rsi > 50:
        return f"RSI = {rsi:.1f}（50 以上），買方力道占優"
    else:
        return f"RSI = {rsi:.1f}（50 以下），賣方力道占優"


def _macd_analysis(subplots: dict) -> str:
    macd_line = subplots["macd"]["MACD"].iloc[-1]
    signal = subplots["macd"]["Signal"].iloc[-1]
    hist = subplots["macd"]["Histogram"].iloc[-1]

    if pd.isna(macd_line) or pd.isna(signal):
        return "MACD 數據不足"
    parts = []
    if macd_line > signal:
        parts.append("MACD 在訊號線上方（多頭）")
    else:
        parts.append("MACD 在訊號線下方（空頭）")
    if hist > 0:
        parts.append("柱狀圖正值，動能增強")
    else:
        parts.append("柱狀圖負值，動能減弱")
    return "，".join(parts)


def _stoch_analysis(subplots: dict) -> str:
    k = subplots["stoch"]["%K"].iloc[-1]
    d = subplots["stoch"]["%D"].iloc[-1]
    if pd.isna(k) or pd.isna(d):
        return "Stochastic 數據不足"
    parts = []
    if k > d:
        parts.append("%K 向上突破 %D（偏多交叉）")
    elif k < d:
        parts.append("%K 跌破 %D（偏空交叉）")
    if k > 80:
        parts.append(f"%K = {k:.1f}（超買區）")
    elif k < 20:
        parts.append(f"%K = {k:.1f}（超賣區）")
    else:
        parts.append(f"%K = {k:.1f}")
    return "，".join(parts)


def _obv_analysis(overlay: pd.DataFrame, subplots: dict) -> str:
    obv = subplots["obv"]["OBV"]
    close = overlay["close"]
    if len(obv) < 2 or len(close) < 2:
        return "OBV 數據不足"
    obv_trend = "上升" if obv.iloc[-1] > obv.iloc[-len(obv) // 2] else "下降"
    price_trend = "上升" if close.iloc[-1] > close.iloc[-len(close) // 2] else "下降"
    if obv_trend == price_trend:
        return f"OBV 與價格同步{obv_trend}，量價關係健康"
    else:
        return f"OBV {obv_trend}但價格{price_trend}，出現背離訊號，可能趨勢反轉"


def _volatility_analysis(overlay: pd.DataFrame) -> str:
    bb_width = overlay["BB_upper"].iloc[-1] - overlay["BB_lower"].iloc[-1]
    avg_width = (overlay["BB_upper"] - overlay["BB_lower"]).mean()
    close = overlay["close"].iloc[-1]
    bb_pct = (bb_width / close) * 100
    if pd.isna(bb_width):
        return "布林通道數據不足"
    if bb_width > avg_width * 1.2:
        return f"波動率偏高（通道寬度 {bb_pct:.1f}%），價格波動加劇"
    elif bb_width < avg_width * 0.8:
        return f"波動率偏低（通道寬度 {bb_pct:.1f}%），可能即將突破"
    else:
        return f"波動率正常（通道寬度 {bb_pct:.1f}%）"


def _funding_analysis(fr: dict) -> str:
    rate = fr["rate"]
    pct = fr["rate_pct"]
    if rate > 0.0001:
        return f"資金費率 {pct:+.4f}%（基準：>+0.01% 偏高），正值偏高，Long 持倉成本增加，市場情緒偏亢奮"
    elif rate > 0:
        return f"資金費率 {pct:+.4f}% （基準：+0.01%~0 正常），微正值，市場略偏多"
    elif rate > -0.0001:
        return f"資金費率 {pct:+.4f}%（基準：0~-0.01% 正常），微負值，市場略偏空"
    else:
        return f"資金費率 {pct:+.4f}%（基準：<-0.01% 偏低），負值偏高，Short 成本增加，市場情緒偏悲觀"


def _oi_analysis(oi: dict) -> str:
    val = oi["open_interest"]
    chg = oi["oi_change_pct"]
    display = f"{val:.0f}" if val < 1e6 else f"{val/1e6:.2f}M"
    if chg is None:
        return f"OI {display}，無歷史趨勢數據"
    if chg > 3:
        return f"OI {display}，近 12h +{chg}%，顯著增加，市場參與度升溫"
    elif chg > 0:
        return f"OI {display}，近 12h +{chg}%，微幅增加，動能延續"
    elif chg > -3:
        return f"OI {display}，近 12h {chg}%，微幅減少，動能趨緩"
    else:
        return f"OI {display}，近 12h {chg}%，顯著減少，資金正在離場"


def _orderbook_analysis(ob: dict) -> str:
    ratio = ob["bid_ask_ratio"]
    bids = ob["bid_volume"]
    asks = ob["ask_volume"]
    if ratio > 1.3:
        return f"買賣掛單比 {ratio:.2f}（基準：>1.3 買方強勢），買單量 ({bids:.0f}) 顯著大於賣單 ({asks:.0f})，即時支撐較強"
    elif ratio > 1:
        return f"買賣掛單比 {ratio:.2f}（基準：1~1.3 買方略優），買單略多於賣單，即時情緒偏多"
    elif ratio > 0.7:
        return f"買賣掛單比 {ratio:.2f}（基準：0.7~1 賣方略優），賣單略多於買單，即時情緒偏空"
    else:
        return f"買賣掛單比 {ratio:.2f}（基準：<0.7 賣方強勢），賣單量 ({asks:.0f}) 顯著大於買單 ({bids:.0f})，即時壓力較強"


def generate_market_summary(overlay: pd.DataFrame, subplots: dict,
                            sentiment: Optional[dict] = None) -> str:
    sections = []
    sections.append("【趨勢判斷】" + _ma_trend(overlay))
    sections.append("【RSI 動能】" + _rsi_analysis(subplots))
    sections.append("【MACD 訊號】" + _macd_analysis(subplots))
    sections.append("【隨機指標】" + _stoch_analysis(subplots))
    sections.append("【量價關係】" + _obv_analysis(overlay, subplots))
    sections.append("【波動率】" + _volatility_analysis(overlay))
    if sentiment:
        sections.append("【資金費率】" + _funding_analysis(sentiment["funding"]))
        sections.append("【未平倉量】" + _oi_analysis(sentiment["oi"]))
        sections.append("【掛單深度】" + _orderbook_analysis(sentiment["orderbook"]))
    extra = []
    if sentiment:
        extra.append({"label": "資金費率 > 0", "bullish": sentiment["funding"]["rate"] > 0})
        extra.append({"label": "掛單比 > 1", "bullish": sentiment["orderbook"]["bid_ask_ratio"] > 1})
        if sentiment["oi"]["oi_change_pct"] is not None:
            extra.append({"label": "OI 12h 增", "bullish": sentiment["oi"]["oi_change_pct"] > 0})
    sections.append("【綜合建議】" + _overall_suggestion(overlay, subplots, extra_signals=extra if extra else None))
    return "\n\n".join(sections)


def _overall_suggestion(overlay: pd.DataFrame, subplots: dict,
                        extra_signals: Optional[list] = None) -> str:
    signals = analyze_signals(overlay, subplots, extra_signals=extra_signals)
    b = signals["bullish_count"]
    c = signals["bearish_count"]

    if b >= 3:
        direction = "偏多"
        advice = "可考慮逢低布局，留意 RSI 是否進入超買區"
    elif c >= 3:
        direction = "偏空"
        advice = "建議謹慎觀望，等待止跌訊號出現"
    else:
        direction = "震盪"
        advice = "方向不明確，建議降低倉位或等待明確突破訊號"

    return f"綜合 {b} 多 / {c} 空訊號，市場情緒{direction}。{advice}。"


def analyze_signals(overlay: pd.DataFrame, subplots: dict,
                    extra_signals: Optional[list] = None) -> dict:
    close = overlay["close"].iloc[-1]
    sma20 = overlay["SMA_20"].iloc[-1]
    sma50 = overlay["SMA_50"].iloc[-1]
    rsi = subplots["rsi"]["RSI"].iloc[-1]
    macd_line = subplots["macd"]["MACD"].iloc[-1]
    signal = subplots["macd"]["Signal"].iloc[-1]

    items = [
        {"label": "收盤 > SMA 20", "bullish": bool(close > sma20)},
        {"label": "SMA 20 > SMA 50", "bullish": bool(sma20 > sma50)},
        {"label": "RSI > 50", "bullish": bool(rsi > 50)},
        {"label": "MACD > 訊號線", "bullish": bool(macd_line > signal)},
    ]
    if extra_signals:
        items.extend(extra_signals)

    b = sum(1 for i in items if i["bullish"])
    c = len(items) - b

    if b >= round(len(items) * 0.75):
        direction = "偏多"
    elif c >= round(len(items) * 0.75):
        direction = "偏空"
    else:
        direction = "震盪"

    return {
        "items": items,
        "bullish_count": b,
        "bearish_count": c,
        "total": len(items),
        "direction": direction,
    }
