import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Optional
from analysis.support_resistance import calc_swing_levels


def _add_color_legend(fig: go.Figure, panel_name: str, row: int):
    yaxis_key = f"yaxis{row}" if row > 1 else "yaxis"
    domain = fig.layout[yaxis_key].domain
    specs = {
        "price": [
            ("orange", "SMA 20"),
            ("green", "SMA 50"),
            ("red", "SMA 200"),
            ("purple", "EMA 12"),
            ("brown", "EMA 26"),
            ("gray", "BB"),
        ],
        "volume": [("lightblue", "Volume")],
        "rsi": [("#4FC3F7", "RSI")],
        "macd": [
            ("#4FC3F7", "MACD"),
            ("orange", "Signal"),
            ("green", "Histogram"),
        ],
        "stoch": [("orange", "%D"), ("#4FC3F7", "%K")],
        "obv": [("purple", "OBV")],
    }
    items = specs.get(panel_name, [])
    parts = [f"<span style='color:{c}'>─ {n}</span>" for c, n in items]
    fig.add_annotation(
        xref="paper", yref="paper",
        x=1, y=domain[1],
        xanchor="right", yanchor="top",
        text="&nbsp;&nbsp;".join(parts),
        showarrow=False,
        font=dict(size=13),
        bgcolor="rgba(14,17,23,0.75)",
        bordercolor="rgba(255,255,255,0.1)",
        borderwidth=1,
        borderpad=4,
    )

_REQUIRED_OVERLAY_COLS = {"open", "high", "low", "close", "volume",
                          "SMA_20", "SMA_50", "SMA_200",
                          "EMA_12", "EMA_26",
                          "BB_upper", "BB_middle", "BB_lower"}


def _get_active_panels(subplots: dict, show_rsi=True, show_macd=True,
                       show_stoch=True, show_obv=True) -> list:
    """Build ordered list of (panel_name, dataframe) based on toggle flags.

    Volume is always included as the first panel. Indicator panels are
    appended if the corresponding show_ flag is True and data exists.
    """
    panels = [("volume", None)]
    if show_rsi and "rsi" in subplots:
        panels.append(("rsi", subplots["rsi"]))
    if show_macd and "macd" in subplots:
        panels.append(("macd", subplots["macd"]))
    if show_stoch and "stoch" in subplots:
        panels.append(("stoch", subplots["stoch"]))
    if show_obv and "obv" in subplots:
        panels.append(("obv", subplots["obv"]))
    return panels


def build_sr_chart(overlay: pd.DataFrame, levels: dict, lookback: int = 40) -> go.Figure:
    df = overlay.iloc[-lookback:].copy()
    pivot = levels["pivot_points"]

    swings = calc_swing_levels(df)

    close_min = df["close"].min()
    close_max = df["close"].max()
    y_pad = (close_max - close_min) * 0.15

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        showlegend=False, name="Price",
    ))

    sr_items = [
        ("R3", pivot["R3"], "#ff6b6b"),
        ("R2", pivot["R2"], "#ff8787"),
        ("R1", pivot["R1"], "#ffa8a8"),
        ("S1", pivot["S1"], "#69db7c"),
        ("S2", pivot["S2"], "#51cf66"),
        ("S3", pivot["S3"], "#40c057"),
    ]

    for label, val, color in sr_items:
        if val < close_min - y_pad or val > close_max + y_pad:
            continue
        fig.add_hline(y=val, line=dict(color=color, width=1, dash="dash"))
        fig.add_annotation(x=df.index[-1], y=val, text=label,
                           xanchor="right", yanchor="bottom",
                           font=dict(size=11, color=color), showarrow=False)

    for i, sh in enumerate(swings["recent_resistances"][:3]):
        if sh < close_min - y_pad or sh > close_max + y_pad:
            continue
        fig.add_hline(y=sh, line=dict(color="#ffa8a8", width=1, dash="dot"))
        fig.add_annotation(x=df.index[0], y=sh, text=f"H{i+1}",
                           xanchor="left", yanchor="bottom",
                           font=dict(size=10, color="#ffa8a8"), showarrow=False)
    for i, sl in enumerate(swings["recent_supports"][:3]):
        if sl < close_min - y_pad or sl > close_max + y_pad:
            continue
        fig.add_hline(y=sl, line=dict(color="#69db7c", width=1, dash="dot"))
        fig.add_annotation(x=df.index[0], y=sl, text=f"L{i+1}",
                           xanchor="left", yanchor="top",
                           font=dict(size=10, color="#69db7c"), showarrow=False)

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=10, b=20),
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        showlegend=False,
        yaxis=dict(fixedrange=False),
    )
    return fig


def build_chart(overlay: pd.DataFrame, subplots: dict,
                show_sma=True, show_ema=True, show_bb=True,
                show_rsi=True, show_macd=True,
                show_stoch=True, show_obv=True) -> go.Figure:
    """Build a multi-panel Plotly figure from OHLCV overlay and indicator data.

    Args:
        overlay: DataFrame with columns open, high, low, close, volume,
                 SMA_20/50/200, EMA_12/26, BB_upper/middle/lower.
        subplots: dict with keys "rsi", "macd", "stoch", "obv" mapping to
                  DataFrames with the indicator's expected columns.
        show_sma/ema/bb: toggle overlay indicator lines on the price panel.
        show_rsi/macd/stoch/obv: toggle subplot indicator panels.

    Returns:
        A Plotly Figure with shared-x subplots, ready for Streamlit rendering.
    """
    missing = _REQUIRED_OVERLAY_COLS - set(overlay.columns)
    if missing:
        raise ValueError(f"Overlay missing required columns: {sorted(missing)}")

    panels = _get_active_panels(subplots, show_rsi, show_macd, show_stoch, show_obv)
    n_panels = len(panels)
    row_heights = [0.4] + [0.15] * n_panels

    fig = make_subplots(
        rows=1 + n_panels, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=row_heights,
    )

    # Row 1: Candlestick + overlays
    fig.add_trace(go.Candlestick(
        x=overlay.index,
        open=overlay["open"],
        high=overlay["high"],
        low=overlay["low"],
        close=overlay["close"],
        name="Price",
        showlegend=False,
    ), row=1, col=1)

    if show_sma:
        fig.add_trace(go.Scatter(x=overlay.index, y=overlay["SMA_20"],
            line=dict(color="orange", width=1), name="SMA 20",
            showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=overlay.index, y=overlay["SMA_50"],
            line=dict(color="green", width=1), name="SMA 50",
            showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=overlay.index, y=overlay["SMA_200"],
            line=dict(color="red", width=1), name="SMA 200",
            showlegend=False), row=1, col=1)

    if show_ema:
        fig.add_trace(go.Scatter(x=overlay.index, y=overlay["EMA_12"],
            line=dict(color="purple", width=1, dash="dot"), name="EMA 12",
            showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=overlay.index, y=overlay["EMA_26"],
            line=dict(color="brown", width=1, dash="dot"), name="EMA 26",
            showlegend=False), row=1, col=1)

    if show_bb:
        fig.add_trace(go.Scatter(x=overlay.index, y=overlay["BB_upper"],
            line=dict(color="gray", width=1, dash="dash"), name="BB Upper",
            showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=overlay.index, y=overlay["BB_middle"],
            line=dict(color="gray", width=1), name="BB Middle",
            showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=overlay.index, y=overlay["BB_lower"],
            line=dict(color="gray", width=1, dash="dash"), name="BB Lower",
            fill="tonexty", fillcolor="rgba(128,128,128,0.1)",
            showlegend=False), row=1, col=1)

    _add_color_legend(fig, "price", 1)

    current_row = 2

    for panel_name, panel_df in panels:
        if panel_name == "volume":
            fig.add_trace(go.Bar(x=overlay.index, y=overlay["volume"],
                name="Volume", marker_color="lightblue",
                showlegend=False), row=current_row, col=1)

        elif panel_name == "rsi":
            fig.add_trace(go.Scatter(x=panel_df.index, y=panel_df["RSI"],
                line=dict(color="#4FC3F7", width=1), name="RSI",
                showlegend=False), row=current_row, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=current_row, col=1)
            fig.update_yaxes(range=[0, 100], row=current_row, col=1)

        elif panel_name == "macd":
            fig.add_trace(go.Scatter(x=panel_df.index, y=panel_df["MACD"],
                line=dict(color="#4FC3F7", width=1), name="MACD",
                showlegend=False), row=current_row, col=1)
            fig.add_trace(go.Scatter(x=panel_df.index, y=panel_df["Signal"],
                line=dict(color="orange", width=1), name="Signal",
                showlegend=False), row=current_row, col=1)
            colors = ["green" if v >= 0 else "red" for v in panel_df["Histogram"]]
            fig.add_trace(go.Bar(x=panel_df.index, y=panel_df["Histogram"],
                marker_color=colors, name="Histogram",
                showlegend=False), row=current_row, col=1)

        elif panel_name == "stoch":
            fig.add_trace(go.Scatter(x=panel_df.index, y=panel_df["%K"],
                line=dict(color="#4FC3F7", width=1), name="%K",
                showlegend=False), row=current_row, col=1)
            fig.add_trace(go.Scatter(x=panel_df.index, y=panel_df["%D"],
                line=dict(color="orange", width=1), name="%D",
                showlegend=False), row=current_row, col=1)
            fig.update_yaxes(range=[0, 100], row=current_row, col=1)

        elif panel_name == "obv":
            fig.add_trace(go.Scatter(x=panel_df.index, y=panel_df["OBV"],
                line=dict(color="purple", width=1), name="OBV",
                showlegend=False), row=current_row, col=1)

        _add_color_legend(fig, panel_name, current_row)
        current_row += 1

    fig.update_layout(
        height=200 * (1 + n_panels),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        template="plotly_dark",
        margin=dict(l=40, r=40, t=20, b=40),
        showlegend=False,
    )

    return fig
