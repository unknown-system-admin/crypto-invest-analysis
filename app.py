import streamlit as st
import pandas as pd
from data.fetcher import fetch_ohlcv
from data.fear_greed import fetch_fear_greed
from data.market_sentiment import fetch_funding_rate, fetch_open_interest, fetch_orderbook_ratio
from indicators.calculator import compute_all
from charts.plotter import build_chart, build_sr_chart
from analysis.support_resistance import compute_all_levels, calc_swing_levels
from analysis.summary import generate_market_summary, analyze_signals
from analysis.multi_tf import analyze_multi_timeframe
from config import SYMBOLS, TIMEFRAMES, DEFAULT_CANDLE_COUNT

st.set_page_config(layout="wide", page_title="Crypto Invest Analysis")

st.title("Crypto Invest Analysis — Technical Dashboard")

with st.sidebar:
    st.header("Settings")
    symbol = st.selectbox("Symbol", SYMBOLS)
    timeframe = st.selectbox("Timeframe", TIMEFRAMES, index=3)
    limit = st.slider("Candles", 50, 500, DEFAULT_CANDLE_COUNT)
    st.markdown("---")
    st.subheader("Indicators")
    show_sma = st.checkbox("SMA", True,
        help="簡單移動平均線：計算過去 N 期收盤價平均值，平滑價格數據，判斷趨勢方向")
    show_ema = st.checkbox("EMA", True,
        help="指數移動平均線：對近期價格加權更多，反應比 SMA 更靈敏")
    show_bb = st.checkbox("Bollinger Bands", True,
        help="布林通道：中軌 SMA ± 標準差，價格觸及上下軌可能反轉，通道寬度反映波動率")
    show_rsi = st.checkbox("RSI", True,
        help="相對強弱指標：0-100，>70 超買區 / <30 超賣區，判斷潛在反轉點")
    show_macd = st.checkbox("MACD", True,
        help="指數平滑異同平均線：快線慢線差離值與訊號線，判斷趨勢方向與強度")
    show_stoch = st.checkbox("Stochastic", True,
        help="隨機指標：%K 與 %D 交叉產生買賣訊號，>80 超買 / <20 超賣")
    show_obv = st.checkbox("OBV", True,
        help="能量潮指標：成交量與價格結合，確認趨勢強度；背離可能預示反轉")

@st.cache_data(ttl=60)
def load_data(symbol, timeframe, limit):
    df = fetch_ohlcv(symbol, timeframe, limit)
    return df

with st.spinner("Fetching market data..."):
    try:
        df = load_data(symbol, timeframe, limit)
    except (ConnectionError, ValueError, RuntimeError) as e:
        st.error(f"Failed to fetch data: {e}")
        st.stop()

if df.empty:
    st.error("No data returned for the selected symbol and timeframe.")
    st.stop()

result = compute_all(df)
levels = compute_all_levels(df, result["overlay"])
@st.cache_data(ttl=120)
def load_sentiment(sym):
    fr = fetch_funding_rate(sym)
    oi = fetch_open_interest(sym)
    ob = fetch_orderbook_ratio(sym)
    return fr, oi, ob

try:
    fr_data, oi_data, ob_data = load_sentiment(symbol)
except Exception:
    fr_data = oi_data = ob_data = None

sentiment = None
extra = []
if fr_data is not None:
    extra.append({"label": "資金費率 > 0 (偏多)", "bullish": fr_data["rate"] > 0})
    sentiment = {"funding": fr_data, "oi": oi_data, "orderbook": ob_data}
if ob_data is not None:
    extra.append({"label": f"買賣掛單比 > 1", "bullish": ob_data["bid_ask_ratio"] > 1})
if oi_data is not None and oi_data["oi_change_pct"] is not None:
    extra.append({"label": f"OI 12h 增加", "bullish": oi_data["oi_change_pct"] > 0})

summary = generate_market_summary(result["overlay"], result["subplots"], sentiment=sentiment)
signals = analyze_signals(result["overlay"], result["subplots"],
                          extra_signals=extra if extra else None)

fig = build_chart(result["overlay"], result["subplots"],
                  show_sma=show_sma, show_ema=show_ema, show_bb=show_bb,
                  show_rsi=show_rsi, show_macd=show_macd,
                  show_stoch=show_stoch, show_obv=show_obv)
st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🎯 近期壓力支撐")
    sr_fig = build_sr_chart(result["overlay"], levels, lookback=40)
    st.plotly_chart(sr_fig, use_container_width=True)

    pivot = levels["pivot_points"]
    swings = calc_swing_levels(result["overlay"].iloc[-40:])
    res_col, sup_col = st.columns(2)
    with res_col:
        st.markdown("**壓力 R**")
        st.markdown(f"R3：{pivot['R3']:.2f}")
        st.markdown(f"R2：{pivot['R2']:.2f}")
        st.markdown(f"R1：{pivot['R1']:.2f}")
        for i, sh in enumerate(swings["recent_resistances"][:3]):
            st.markdown(f"H{i+1}：{sh:.2f}")
    with sup_col:
        st.markdown("**支撐 S**")
        st.markdown(f"S1：{pivot['S1']:.2f}")
        st.markdown(f"S2：{pivot['S2']:.2f}")
        st.markdown(f"S3：{pivot['S3']:.2f}")
        for i, sl in enumerate(swings["recent_supports"][:3]):
            st.markdown(f"L{i+1}：{sl:.2f}")

    ma = levels["ma_levels"]
    st.markdown("**關鍵均線關卡**")
    ma_cols = st.columns(3)
    items = list(ma.items())
    for i, (name, val) in enumerate(items):
        ma_cols[i % 3].markdown(f"{name}：{val}")

with col2:
    st.subheader("📊 多空訊號儀表板")
    sig = signals
    b, c, t = sig["bullish_count"], sig["bearish_count"], sig["total"]
    bias = (b - c) / t  # -1 ~ +1
    bias_pct = bias * 100

    dir_color = {"偏多": "#28a745", "偏空": "#dc3545", "震盪": "#ffc107"}
    direction = sig["direction"]
    dir_clr = dir_color[direction]

    st.markdown(
        f"<div style='text-align:center;font-size:14px;margin-bottom:8px'>"
        f"<span style='color:#28a745'>▲ 多方 {b}</span>　"
        f"<span style='color:#dc3545'>▼ 空方 {c}</span>　"
        f"<span style='font-weight:bold;color:{dir_clr}'>{direction}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    pointer_left = (bias + 1) / 2 * 100
    gauge_html = f"""
    <div style='position:relative;height:36px;margin:4px 0 16px'>
        <div style='position:absolute;top:8px;left:0;right:0;height:12px;
            border-radius:6px;background:linear-gradient(to right,#dc3545,#ff851b,#ffc107,#28a745);
            box-shadow:inset 0 1px 3px rgba(0,0,0,0.3)'></div>
        <div style='position:absolute;top:0;left:{pointer_left}%;transform:translateX(-50%);
            width:0;height:0;border-left:8px solid transparent;border-right:8px solid transparent;
            border-top:14px solid {dir_clr};filter:drop-shadow(0 1px 2px rgba(0,0,0,0.5))'></div>
        <div style='position:absolute;bottom:0;left:0;font-size:10px;color:#dc3545'>空</div>
        <div style='position:absolute;bottom:0;right:0;font-size:10px;color:#28a745'>多</div>
        <div style='position:absolute;bottom:0;left:50%;transform:translateX(-50%);
            font-size:10px;color:#ffc107'>中立</div>
    </div>
    """
    st.markdown(gauge_html, unsafe_allow_html=True)

    for item in sig["items"]:
        if item["bullish"]:
            icon, clr = "🟢", "#28a745"
        else:
            icon, clr = "🔴", "#dc3545"
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;padding:3px 8px;"
            f"border-bottom:1px solid #2d2d2d;font-size:14px'>"
            f"<span>{item['label']}</span>"
            f"<span style='color:{clr}'>{icon}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        f"<div style='margin-top:12px;padding:8px 12px;border-radius:8px;"
        f"background:{dir_clr}22;border:1px solid {dir_clr}44;"
        f"font-size:13px;line-height:1.5'>{summary}</div>",
        unsafe_allow_html=True,
    )

latest = result["overlay"].iloc[-1]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Close", f"${latest['close']:.2f}")
col2.metric("SMA 20", f"${latest['SMA_20']:.2f}")
col3.metric("RSI", f"{result['subplots']['rsi']['RSI'].iloc[-1]:.1f}")
col4.metric("MACD", f"{result['subplots']['macd']['MACD'].iloc[-1]:.2f}")

st.markdown("---")
st.subheader("📈 跨級別趨勢分析")

with st.spinner("分析多時間框架..."):
    try:
        tf_results = analyze_multi_timeframe(symbol)
    except Exception as e:
        st.warning(f"跨級別分析失敗：{e}")
        tf_results = []

if tf_results:
    dir_color = {"偏多": "#28a745", "偏空": "#dc3545", "震盪": "#ffc107"}
    cols = st.columns(len(tf_results))
    for i, r in enumerate(tf_results):
        with cols[i]:
            if r["error"]:
                st.markdown(f"**{r['label']}**\n\n❌ 資料錯誤")
                continue
            clr = dir_color.get(r["direction"], "#ffc107")
            st.markdown(
                f"<div style='background:#1a1d23;border-radius:8px;padding:10px;"
                f"border:1px solid {clr}44;text-align:center'>"
                f"<div style='font-weight:bold;font-size:16px;margin-bottom:6px'>{r['label']}</div>"
                f"<div style='font-size:20px;font-weight:bold;color:{clr}'>{r['direction']}</div>"
                f"<div style='font-size:12px;margin-top:6px'>"
                f"<span style='color:#28a745'>▲{r['bullish']}</span> "
                f"<span style='color:#dc3545'>▼{r['bearish']}</span>"
                f"</div>"
                f"<div style='font-size:12px;color:#aaa;margin-top:4px'>"
                f"RSI {r['rsi']:.1f} | ${r['close']:.0f}"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
    st.caption("跨 1h / 4h / 1d / 1w 綜合比較，自動抓取各級別 200 根 K 線計算")

st.markdown("---")
st.subheader("😨 恐懼與貪婪指數")

@st.cache_data(ttl=3600)
def load_fng():
    return fetch_fear_greed(limit=1)

try:
    fng = load_fng()
    if fng:
        item = fng[0]
        val = item["value"]
        label = item["value_classification"]

        if val <= 24:
            color = "#dc3545"
            emoji = "😱"
        elif val <= 44:
            color = "#ff851b"
            emoji = "😟"
        elif val <= 55:
            color = "#ffc107"
            emoji = "😐"
        elif val <= 74:
            color = "#28a745"
            emoji = "😊"
        else:
            color = "#17a2b8"
            emoji = "🤩"

        fg_col1, fg_col2 = st.columns([1, 3])
        with fg_col1:
            st.markdown(
                f"<div style='text-align:center;font-size:48px;font-weight:bold;color:{color}'>{val}</div>"
                f"<div style='text-align:center;font-size:16px;color:{color}'>{emoji} {label}</div>",
                unsafe_allow_html=True,
            )
        with fg_col2:
            pass
        st.caption(f"更新時間：{item['timestamp'].strftime('%Y-%m-%d %H:%M')}")
except (ConnectionError, ValueError) as e:
    st.warning(f"無法取得恐懼與貪婪指數：{e}")

st.markdown("---")
st.subheader("📊 籌碼面與市場情緒")

if fr_data and oi_data and ob_data:
    sc1, sc2, sc3 = st.columns(3)

    with sc1:
        fr_color = "#28a745" if fr_data["rate"] > 0 else "#dc3545"
        fr_label = "多頭偏多 (Long 付費)" if fr_data["rate"] > 0 else "空頭偏多 (Short 付費)"
        fr_range = "正常 0~±0.01%"
        if abs(fr_data["rate"]) > 0.0001:
            fr_range = "⚠ 偏高 >±0.01%"
        st.markdown(
            f"<div style='background:#1a1d23;border-radius:8px;padding:12px;text-align:center'>"
            f"<div style='font-size:12px;color:#aaa'>資金費率 <span style='color:#666;font-size:10px'>（{fr_range}）</span></div>"
            f"<div style='font-size:22px;font-weight:bold;color:{fr_color}'>{fr_data['rate_pct']:+.4f}%</div>"
            f"<div style='font-size:12px;color:{fr_color}'>{fr_label}</div>"
            f"</div>", unsafe_allow_html=True,
        )

    with sc2:
        oi_val = oi_data["open_interest"]
        oi_display = f"{oi_val:.0f}" if oi_val < 1e6 else f"{oi_val/1e6:.2f}M"
        oi_chg = oi_data["oi_change_pct"]
        if oi_chg is not None:
            oi_chg_color = "#28a745" if oi_chg > 0 else "#dc3545"
            oi_chg_str = f"+{oi_chg}%" if oi_chg > 0 else f"{oi_chg}%"
            oi_analysis = f"近 12h <span style='color:{oi_chg_color}'>{oi_chg_str}</span>"
        else:
            oi_analysis = "無歷史資料"
        st.markdown(
            f"<div style='background:#1a1d23;border-radius:8px;padding:12px;text-align:center'>"
            f"<div style='font-size:12px;color:#aaa'>未平倉量 (OI)</div>"
            f"<div style='font-size:22px;font-weight:bold;color:#4FC3F7'>{oi_display}</div>"
            f"<div style='font-size:12px;color:#aaa'>{symbol.split('/')[0]} 合約</div>"
            f"<div style='font-size:11px;margin-top:4px'>{oi_analysis}</div>"
            f"</div>", unsafe_allow_html=True,
        )

    with sc3:
        ratio = ob_data["bid_ask_ratio"]
        if ratio > 1.2:
            r_color, r_label = "#28a745", "買單強勢"
            r_range = ">1.3 強 / 1~1.3 略優"
        elif ratio < 0.8:
            r_color, r_label = "#dc3545", "賣單強勢"
            r_range = "<0.7 強 / 0.7~1 略優"
        else:
            r_color, r_label = "#ffc107", "買賣均衡"
            r_range = "正常 0.7~1.3"
        st.markdown(
            f"<div style='background:#1a1d23;border-radius:8px;padding:12px;text-align:center'>"
            f"<div style='font-size:12px;color:#aaa'>買賣掛單比 <span style='color:#666;font-size:10px'>（{r_range}）</span></div>"
            f"<div style='font-size:22px;font-weight:bold;color:{r_color}'>{ratio:.2f}</div>"
            f"<div style='font-size:12px;color:{r_color}'>{r_label}</div>"
            f"<div style='font-size:10px;color:#666'>買 {ob_data['bid_volume']:.0f} / 賣 {ob_data['ask_volume']:.0f}</div>"
            f"</div>", unsafe_allow_html=True,
        )
