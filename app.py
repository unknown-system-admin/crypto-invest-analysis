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

PAGE_NAMES = {
    "dashboard": "📈 技術分析儀表板",
    "backtest": "📊 策略回測",
    "portfolio": "💰 交易 Dashboard",
}

if "page" not in st.session_state:
    st.session_state.page = "dashboard"

def switch_page(name):
    st.session_state.page = name

with st.sidebar:
    st.title("Navigation")
    for key, label in PAGE_NAMES.items():
        if st.button(label, use_container_width=True,
                     type="primary" if st.session_state.page == key else "secondary"):
            switch_page(key)
    st.markdown("---")

def dashboard():
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
        bias = (b - c) / t

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

def backtest_page():
    tab_mode = st.radio("模式", ["單次回測", "⚡ 參數優化"], horizontal=True, label_visibility="collapsed")
    if tab_mode == "單次回測":
        _single_backtest_ui()
    else:
        _optimizer_ui()

def _single_backtest_ui():
    st.title("📊 策略回測")
    st.markdown("---")

    from trading.backtest import run_backtest
    from trading.config import DEFAULT_STRATEGIES, DEFAULT_RISK

    with st.sidebar:
        st.header("回測參數")
        symbol = st.selectbox("交易對", ["BTC/USDT", "SOL/USDT"], key="bt_symbol")
        tf = st.selectbox("時間框架", ["1h", "4h", "1d"], index=0, key="bt_tf")
        lookback = st.slider("回測K線數", 100, 2000, 500, key="bt_lookback")
        capital = st.number_input("初始資金 ($)", 1000, 100000, 10000, step=1000, key="bt_capital")
        fee = st.number_input("手續費率", 0.0, 1.0, 0.1, step=0.05, format="%.3f", key="bt_fee") / 100
        slippage = st.number_input("滑價", 0.0, 1.0, 0.05, step=0.01, format="%.3f", key="bt_slip") / 100

        st.subheader("策略設定")
        enabled_strategies = {}
        for sname, scfg in DEFAULT_STRATEGIES.items():
            enabled = st.checkbox(f"{sname}", value=scfg["enabled"], key=f"bt_{sname}")
            enabled_strategies[sname] = {
                "enabled": enabled,
                "params": scfg["params"],
                "weight": scfg["weight"],
            }

        run_btn = st.button("🚀 開始回測", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner("回測進行中..."):
            try:
                result = run_backtest(
                    symbol=symbol, timeframe=tf, limit=lookback,
                    initial_capital=float(capital),
                    fee_rate=fee, slippage=slippage,
                    strategy_configs=enabled_strategies,
                    signal_threshold=0.6,
                    risk_config=DEFAULT_RISK,
                )
            except Exception as e:
                st.error(f"回測失敗: {e}")
                return

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("總報酬率", f"{result.total_return_pct:+.2f}%")
        with col2:
            st.metric("CAGR", f"{result.cagr:+.2f}%")
        with col3:
            st.metric("最大回撤", f"{result.max_drawdown_pct:.2f}%",
                      delta_color="inverse")
        with col4:
            st.metric("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")

        col5, col6, col7, col8 = st.columns(4)
        with col5:
            st.metric("勝率", f"{result.win_rate:.1f}%")
        with col6:
            st.metric("交易次數", result.total_trades)
        with col7:
            st.metric("初始資金", f"${result.initial_capital:,.0f}")
        with col8:
            st.metric("最終權益", f"${result.final_equity:,.2f}")

        st.subheader("權益曲線")
        if result.equity_curve:
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=result.equity_curve,
                mode="lines",
                name="Equity",
                line=dict(color="#00FF88", width=2),
                fill="tozeroy",
            ))
            fig.update_layout(
                height=400, template="plotly_dark",
                yaxis_title="Equity ($)",
                xaxis_title="Bar",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("每月報酬率")
        if result.monthly_returns:
            months = list(result.monthly_returns.keys())
            rets = list(result.monthly_returns.values())
            import plotly.express as px
            fig2 = px.bar(x=months, y=rets, color=rets,
                          color_continuous_scale=["#dc3545", "#ffc107", "#28a745"])
            fig2.update_layout(
                height=250, template="plotly_dark",
                xaxis_title="Month", yaxis_title="Return %",
                showlegend=False,
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("交易明細")
        if result.trades:
            st.dataframe(pd.DataFrame(result.trades), use_container_width=True)
        else:
            st.info("回測期間無交易訊號")
    else:
        st.info("請在左側設定參數後點擊「開始回測」")

def _optimizer_ui():
    st.title("⚡ 參數優化")
    opt_mode = st.radio("優化模式", ["權重掃描", "RSI 參數掃描", "MA 參數掃描"], horizontal=True, label_visibility="collapsed")
    if opt_mode == "權重掃描":
        _weight_sweep_ui()
    elif opt_mode == "RSI 參數掃描":
        _rsi_sweep_ui()
    else:
        _ma_sweep_ui()


def _weight_sweep_ui():
    st.markdown("---")

    from trading.optimizer import ParamSweepConfig, SweepResult, run_parameter_sweep
    from trading.optimizer import save_sweep_results
    from trading.config import DEFAULT_STRATEGIES

    with st.sidebar:
        st.header("優化設定")
        symbol = st.selectbox("交易對", ["BTC/USDT", "SOL/USDT"], key="opt_symbol")
        tf = st.selectbox("時間框架", ["1h", "4h", "1d"], index=0, key="opt_tf")
        lookback = st.slider("回測K線數", 200, 2000, 500, key="opt_lookback")
        capital = st.number_input("初始資金 ($)", 1000, 100000, 10000, step=1000, key="opt_capital")

        st.subheader("權重範圍 (0~3)")
        w_ma = st.slider("MA Cross", 0, 3, (0, 3), key="opt_w_ma")
        w_rsi = st.slider("RSI", 0, 3, (0, 3), key="opt_w_rsi")
        w_macd = st.slider("MACD", 0, 3, (0, 3), key="opt_w_macd")
        w_comp = st.slider("Composite", 0, 3, (0, 3), key="opt_w_comp")

        st.subheader("Threshold 範圍")
        th_min, th_max = st.slider("Threshold", 0.3, 0.9, (0.3, 0.7), step=0.1, key="opt_th")
        th_step = st.selectbox("Step", [0.1, 0.2], index=0, key="opt_th_step")

        start_btn = st.button("🚀 開始優化", type="primary", use_container_width=True)

    if start_btn:
        config = ParamSweepConfig(
            symbol=symbol, timeframe=tf, limit=lookback,
            initial_capital=float(capital),
        )
        config.weight_min = min(w_ma[0], w_rsi[0], w_macd[0], w_comp[0])
        config.weight_max = max(w_ma[1], w_rsi[1], w_macd[1], w_comp[1])
        config.threshold_start = th_min
        config.threshold_stop = th_max
        config.threshold_step = th_step

        weight_values = config.weight_max - config.weight_min + 1
        n_strats = len(config.strategy_order)
        n_th = int((th_max - th_min) / th_step) + 1
        total_combos = (weight_values ** n_strats) * n_th
        st.info(f"測試組合共 **{total_combos:,}** 組，預計 30~90 秒")

        progress_bar = st.progress(0, text="初始化...")
        best_text = st.empty()

        all_results = []

        def on_progress(current, total, latest):
            pct = current / total
            progress_bar.progress(pct, text=f"{current}/{total}")
            if latest and latest.total_return_pct > -999:
                all_results.append(latest)
                best = max(all_results, key=lambda r: r.total_return_pct)
                best_text.text(f"當前最佳報酬: {best.total_return_pct:+.2f}%  "
                               f"(Sharpe: {best.sharpe_ratio:.2f}  MaxDD: {best.max_drawdown_pct:.2f}%)")

        try:
            results = run_parameter_sweep(config, progress_callback=on_progress)
        except Exception as e:
            st.error(f"優化失敗: {e}")
            return

        progress_bar.empty()
        st.success(f"✅ 完成！共測試 {len(results):,} 組")
        save_path = save_sweep_results(results, config)
        st.caption(f"結果已儲存: {save_path}")
        _display_optimizer_results(results, config)


def _rsi_sweep_ui():
    st.markdown("---")
    from trading.optimizer import RsiSweepConfig, RsiSweepResult, run_rsi_sweep, save_sweep_results

    with st.sidebar:
        st.header("RSI 參數設定")
        symbol = st.selectbox("交易對", ["BTC/USDT", "SOL/USDT"], key="rsi_symbol")
        tf = st.selectbox("時間框架", ["1h", "4h", "1d"], index=1, key="rsi_tf")
        lookback = st.slider("回測K線數", 200, 2000, 500, key="rsi_lookback")
        capital = st.number_input("初始資金 ($)", 1000, 100000, 10000, step=1000, key="rsi_capital")

        st.subheader("Period")
        p_min, p_max = st.select_slider("Period", options=[7, 9, 11, 13, 14, 15, 17, 19, 21],
                                         value=(7, 21), key="rsi_period")

        st.subheader("Overbought")
        ob_min, ob_max = st.select_slider("Overbought (> Oversold)", options=[65, 70, 75, 80, 85],
                                           value=(65, 85), key="rsi_ob")

        st.subheader("Oversold")
        os_min, os_max = st.select_slider("Oversold (< Overbought)", options=[15, 20, 25, 30, 35],
                                           value=(15, 35), key="rsi_os")

        start_btn = st.button("🚀 開始 RSI 優化", type="primary", use_container_width=True)

    if start_btn:
        period_list = [v for v in [7, 9, 11, 13, 14, 15, 17, 19, 21] if p_min <= v <= p_max]
        ob_list = [v for v in [65, 70, 75, 80, 85] if ob_min <= v <= ob_max]
        os_list = [v for v in [15, 20, 25, 30, 35] if os_min <= v <= os_max]

        config = RsiSweepConfig(
            symbol=symbol, timeframe=tf, limit=lookback,
            initial_capital=float(capital),
            period_values=period_list, ob_values=ob_list, os_values=os_list,
        )

        total_combos = sum(1 for p in period_list for ob in ob_list for os in os_list if os < ob)
        st.info(f"測試組合共 **{total_combos:,}** 組")

        progress_bar = st.progress(0, text="初始化...")
        best_text = st.empty()
        all_results = []

        def on_progress(cur, total, latest):
            progress_bar.progress(cur / total, text=f"{cur}/{total}")
            if latest:
                all_results.append(latest)
                best = max(all_results, key=lambda r: r.total_return_pct)
                best_text.text(f"當前最佳報酬: {best.total_return_pct:+.2f}%  "
                               f"(Sharpe: {best.sharpe_ratio:.2f}  DD: {best.max_drawdown_pct:.1f}%)")

        try:
            results = run_rsi_sweep(config, on_progress)
        except Exception as e:
            st.error(f"優化失敗: {e}")
            return

        progress_bar.empty()
        st.success(f"✅ 完成！共測試 {len(results):,} 組")
        save_path = save_sweep_results(results, config)
        st.caption(f"結果已儲存: {save_path}")
        _display_rsi_results(results, config)


def _ma_sweep_ui():
    st.markdown("---")
    from trading.optimizer import MaSweepConfig, MaSweepResult, run_ma_sweep, save_sweep_results

    with st.sidebar:
        st.header("MA 參數設定")
        symbol = st.selectbox("交易對", ["BTC/USDT", "SOL/USDT"], key="ma_symbol")
        tf = st.selectbox("時間框架", ["1h", "4h", "1d"], index=1, key="ma_tf")
        lookback = st.slider("回測K線數", 200, 2000, 500, key="ma_lookback")
        capital = st.number_input("初始資金 ($)", 1000, 100000, 10000, step=1000, key="ma_capital")

        st.subheader("快線 (Fast MA)")
        f_min, f_max = st.select_slider("Fast Period", options=[5, 8, 10, 12, 15, 20],
                                         value=(5, 20), key="ma_fast")

        st.subheader("慢線 (Slow MA > Fast)")
        s_min, s_max = st.select_slider("Slow Period", options=[20, 26, 30, 40, 50, 60],
                                         value=(20, 60), key="ma_slow")

        start_btn = st.button("🚀 開始 MA 優化", type="primary", use_container_width=True)

    if start_btn:
        fast_list = [v for v in [5, 8, 10, 12, 15, 20] if f_min <= v <= f_max]
        slow_list = [v for v in [20, 26, 30, 40, 50, 60] if s_min <= v <= s_max]

        config = MaSweepConfig(
            symbol=symbol, timeframe=tf, limit=lookback,
            initial_capital=float(capital),
            fast_values=fast_list, slow_values=slow_list,
        )

        total_combos = sum(1 for f in fast_list for s in slow_list if f < s)
        st.info(f"測試組合共 **{total_combos:,}** 組")

        progress_bar = st.progress(0, text="初始化...")
        best_text = st.empty()
        all_results = []

        def on_progress(cur, total, latest):
            progress_bar.progress(cur / total, text=f"{cur}/{total}")
            if latest:
                all_results.append(latest)
                best = max(all_results, key=lambda r: r.total_return_pct)
                best_text.text(f"當前最佳報酬: {best.total_return_pct:+.2f}%  "
                               f"(Sharpe: {best.sharpe_ratio:.2f}  DD: {best.max_drawdown_pct:.1f}%)")

        try:
            results = run_ma_sweep(config, on_progress)
        except Exception as e:
            st.error(f"優化失敗: {e}")
            return

        progress_bar.empty()
        st.success(f"✅ 完成！共測試 {len(results):,} 組")
        save_path = save_sweep_results(results, config)
        st.caption(f"結果已儲存: {save_path}")
        _display_ma_results(results, config)


def _display_optimizer_results(results, config):
    st.subheader("🏆 Top 30 最佳參數")
    top = results[:30]
    rows = []
    for i, r in enumerate(top):
        p = r.params
        rows.append({
            "排名": i + 1,
            "MA": p.get("ma_cross_weight", "-"),
            "RSI": p.get("rsi_weight", "-"),
            "MACD": p.get("macd_weight", "-"),
            "Comp": p.get("composite_weight", "-"),
            "Thresh": p.get("threshold", "-"),
            "報酬率": f"{r.total_return_pct:+.2f}%",
            "Sharpe": f"{r.sharpe_ratio:.2f}",
            "MaxDD": f"{r.max_drawdown_pct:.2f}%",
            "勝率": f"{r.win_rate:.1f}%",
            "交易": r.total_trades,
        })
    df_top = pd.DataFrame(rows)
    st.dataframe(df_top, use_container_width=True, hide_index=True)

    st.subheader("最佳參數詳情")
    st.json(results[0].to_dict())

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ 套用最佳參數", type="primary", use_container_width=True):
            _apply_best_params(results[0], config)
    with col2:
        st.download_button(
            "📥 下載完整結果 CSV",
            data=pd.DataFrame([r.to_dict() for r in results]).to_csv(index=False).encode(),
            file_name=f"optimization_{config.symbol.replace('/','_')}_{config.timeframe}.csv",
            mime="text/csv", use_container_width=True,
        )


def _apply_best_params(best, config):
    import yaml
    from pathlib import Path

    config_path = Path(__file__).parent / "monitor" / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    strategies = cfg.setdefault("trading", {}).setdefault("strategies", {})
    p = best.params
    for sname in config.strategy_order:
        w = int(p.get(f"{sname}_weight", 0))
        if sname in strategies:
            strategies[sname]["weight"] = w
            strategies[sname]["enabled"] = w > 0

    cfg["trading"]["risk"]["signal_threshold"] = p.get("threshold", 0.6)

    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    st.success("✅ 最佳參數已套用至 monitor/config.yaml")


def _display_rsi_results(results, config):
    st.subheader("🏆 Top 30 最佳參數")
    top = results[:30]
    rows = []
    for i, r in enumerate(top):
        p = r.params
        rows.append({
            "排名": i + 1,
            "Period": p.get("period", "-"),
            "Overbought": p.get("overbought", "-"),
            "Oversold": p.get("oversold", "-"),
            "報酬率": f"{r.total_return_pct:+.2f}%",
            "Sharpe": f"{r.sharpe_ratio:.2f}",
            "MaxDD": f"{r.max_drawdown_pct:.2f}%",
            "勝率": f"{r.win_rate:.1f}%",
            "交易": r.total_trades,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("最佳參數詳情")
    st.json(results[0].to_dict())

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ 套用最佳參數", type="primary", use_container_width=True):
            _apply_rsi_params(results[0])
    with col2:
        st.download_button(
            "📥 下載完整結果 CSV",
            data=pd.DataFrame([r.to_dict() for r in results]).to_csv(index=False).encode(),
            file_name=f"rsi_sweep_{config.symbol.replace('/','_')}_{config.timeframe}.csv",
            mime="text/csv", use_container_width=True,
        )


def _apply_rsi_params(best):
    import yaml
    from pathlib import Path
    config_path = Path(__file__).parent / "monitor" / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    p = best.params
    strategies = cfg.setdefault("trading", {}).setdefault("strategies", {})
    rsi_cfg = strategies.setdefault("rsi", {})
    rsi_cfg["params"] = {"period": p["period"], "overbought": p["overbought"], "oversold": p["oversold"]}
    rsi_cfg["enabled"] = True
    rsi_cfg["weight"] = 1
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    st.success("✅ RSI 參數已套用至 monitor/config.yaml")


def _display_ma_results(results, config):
    st.subheader("🏆 Top 30 最佳參數")
    top = results[:30]
    rows = []
    for i, r in enumerate(top):
        p = r.params
        rows.append({
            "排名": i + 1,
            "Fast": p.get("fast", "-"),
            "Slow": p.get("slow", "-"),
            "報酬率": f"{r.total_return_pct:+.2f}%",
            "Sharpe": f"{r.sharpe_ratio:.2f}",
            "MaxDD": f"{r.max_drawdown_pct:.2f}%",
            "勝率": f"{r.win_rate:.1f}%",
            "交易": r.total_trades,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("最佳參數詳情")
    st.json(results[0].to_dict())

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ 套用最佳參數", type="primary", use_container_width=True):
            _apply_ma_params(results[0])
    with col2:
        st.download_button(
            "📥 下載完整結果 CSV",
            data=pd.DataFrame([r.to_dict() for r in results]).to_csv(index=False).encode(),
            file_name=f"ma_sweep_{config.symbol.replace('/','_')}_{config.timeframe}.csv",
            mime="text/csv", use_container_width=True,
        )


def _apply_ma_params(best):
    import yaml
    from pathlib import Path
    config_path = Path(__file__).parent / "monitor" / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    p = best.params
    strategies = cfg.setdefault("trading", {}).setdefault("strategies", {})
    ma_cfg = strategies.setdefault("ma_cross", {})
    ma_cfg["params"] = {"fast": p["fast"], "slow": p["slow"], "type": "ema"}
    ma_cfg["enabled"] = True
    ma_cfg["weight"] = 1
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    st.success("✅ MA 參數已套用至 monitor/config.yaml")


def portfolio_page():
    st.title("💰 交易 Dashboard")
    st.markdown("---")

    from trading.portfolio import PortfolioStore

    store = PortfolioStore()
    portfolio = store.load()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("現金餘額", f"${portfolio.cash:,.2f}")
    with col2:
        st.metric("市值", f"${portfolio.market_value:,.2f}")
    with col3:
        st.metric("總權益", f"${portfolio.total_equity:,.2f}")
    with col4:
        total_pnl = portfolio.total_pnl
        st.metric("總損益", f"${total_pnl:+,.2f}",
                  delta=f"{total_pnl/portfolio.total_equity*100:+.2f}%" if portfolio.total_equity > 0 else "")

    st.subheader("當前持倉")
    if portfolio.positions:
        pos_data = []
        for p in portfolio.positions:
            pos_data.append({
                "交易對": p.symbol, "方向": p.side, "數量": p.quantity,
                "均價": f"${p.entry_price:,.2f}", "現價": f"${p.current_price:,.2f}",
                "浮盈": f"${p.unrealized_pnl:+,.2f}",
                "ROI%": f"{p.unrealized_pnl_pct:+.2f}%",
            })
        st.dataframe(pos_data, use_container_width=True)
    else:
        st.info("目前無持倉")

    st.subheader("近期訂單")
    if portfolio.orders:
        recent = portfolio.orders[-20:][::-1]
        order_data = []
        for o in recent:
            order_data.append({
                "時間": o.timestamp[-19:], "交易對": o.symbol,
                "方向": o.side, "數量": o.quantity,
                "價格": f"${o.price:,.2f}",
                "損益": f"${o.pnl:+,.2f}" if o.pnl else "-",
            })
        st.dataframe(order_data, use_container_width=True)
    else:
        st.info("暫無訂單記錄")

    st.subheader("帳戶損益趨勢")
    filled_sells = [o for o in portfolio.orders if o.side == "sell" and o.status == "filled"]
    if filled_sells:
        import plotly.graph_objects as go
        cumulative = []
        running = 0.0
        for o in filled_sells:
            running += o.pnl
            cumulative.append(running)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=cumulative, mode="lines",
            name="Cumulative PnL", line=dict(color="#4FC3F7", width=2),
        ))
        fig.update_layout(height=300, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("尚無已平倉交易")

pages = {
    "dashboard": dashboard,
    "backtest": backtest_page,
    "portfolio": portfolio_page,
}
pages[st.session_state.page]()
