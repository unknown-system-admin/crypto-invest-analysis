import streamlit as st
from data.fetcher import fetch_ohlcv
from indicators.calculator import compute_all
from charts.plotter import build_chart
from config import SYMBOLS, TIMEFRAMES, DEFAULT_CANDLE_COUNT

st.set_page_config(layout="wide", page_title="Crypto Invest Analysis")

st.title("Crypto Invest Analysis — Technical Dashboard")

with st.sidebar:
    st.header("Settings")
    symbol = st.selectbox("Symbol", SYMBOLS)
    timeframe = st.selectbox("Timeframe", TIMEFRAMES)
    limit = st.slider("Candles", 50, 500, DEFAULT_CANDLE_COUNT)
    st.markdown("---")
    st.subheader("Indicators")
    show_sma = st.checkbox("SMA", True)
    show_ema = st.checkbox("EMA", True)
    show_bb = st.checkbox("Bollinger Bands", True)
    show_rsi = st.checkbox("RSI", True)
    show_macd = st.checkbox("MACD", True)
    show_stoch = st.checkbox("Stochastic", True)
    show_obv = st.checkbox("OBV", True)

@st.cache_data(ttl=60)
def load_data(symbol, timeframe, limit):
    df = fetch_ohlcv(symbol, timeframe, limit)
    return df

df = load_data(symbol, timeframe, limit)
result = compute_all(df)

fig = build_chart(result["overlay"], result["subplots"],
                  show_sma=show_sma, show_ema=show_ema, show_bb=show_bb,
                  show_rsi=show_rsi, show_macd=show_macd,
                  show_stoch=show_stoch, show_obv=show_obv)
st.plotly_chart(fig, use_container_width=True)

latest = result["overlay"].iloc[-1]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Close", f"${latest['close']:.2f}")
col2.metric("SMA 20", f"${latest['SMA_20']:.2f}")
col3.metric("RSI", f"{result['subplots']['rsi']['RSI'].iloc[-1]:.1f}")
col4.metric("MACD", f"{result['subplots']['macd']['MACD'].iloc[-1]:.2f}")
