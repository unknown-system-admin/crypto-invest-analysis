import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Optional

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
            line=dict(color="orange", width=1), name="SMA 20"), row=1, col=1)
        fig.add_trace(go.Scatter(x=overlay.index, y=overlay["SMA_50"],
            line=dict(color="green", width=1), name="SMA 50"), row=1, col=1)
        fig.add_trace(go.Scatter(x=overlay.index, y=overlay["SMA_200"],
            line=dict(color="red", width=1), name="SMA 200"), row=1, col=1)

    if show_ema:
        fig.add_trace(go.Scatter(x=overlay.index, y=overlay["EMA_12"],
            line=dict(color="purple", width=1, dash="dot"), name="EMA 12"), row=1, col=1)
        fig.add_trace(go.Scatter(x=overlay.index, y=overlay["EMA_26"],
            line=dict(color="brown", width=1, dash="dot"), name="EMA 26"), row=1, col=1)

    if show_bb:
        fig.add_trace(go.Scatter(x=overlay.index, y=overlay["BB_upper"],
            line=dict(color="gray", width=1, dash="dash"), name="BB Upper"), row=1, col=1)
        fig.add_trace(go.Scatter(x=overlay.index, y=overlay["BB_middle"],
            line=dict(color="gray", width=1), name="BB Middle"), row=1, col=1)
        fig.add_trace(go.Scatter(x=overlay.index, y=overlay["BB_lower"],
            line=dict(color="gray", width=1, dash="dash"), name="BB Lower",
            fill="tonexty", fillcolor="rgba(128,128,128,0.1)"), row=1, col=1)

    current_row = 2

    for panel_name, panel_df in panels:
        if panel_name == "volume":
            fig.add_trace(go.Bar(x=overlay.index, y=overlay["volume"],
                name="Volume", marker_color="lightblue"), row=current_row, col=1)

        elif panel_name == "rsi":
            fig.add_trace(go.Scatter(x=panel_df.index, y=panel_df["RSI"],
                line=dict(color="blue", width=1), name="RSI"), row=current_row, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=current_row, col=1)
            fig.update_yaxes(range=[0, 100], row=current_row, col=1)

        elif panel_name == "macd":
            fig.add_trace(go.Scatter(x=panel_df.index, y=panel_df["MACD"],
                line=dict(color="blue", width=1), name="MACD"), row=current_row, col=1)
            fig.add_trace(go.Scatter(x=panel_df.index, y=panel_df["Signal"],
                line=dict(color="orange", width=1), name="Signal"), row=current_row, col=1)
            colors = ["green" if v >= 0 else "red" for v in panel_df["Histogram"]]
            fig.add_trace(go.Bar(x=panel_df.index, y=panel_df["Histogram"],
                marker_color=colors, name="Histogram"), row=current_row, col=1)

        elif panel_name == "stoch":
            fig.add_trace(go.Scatter(x=panel_df.index, y=panel_df["%K"],
                line=dict(color="blue", width=1), name="%K"), row=current_row, col=1)
            fig.add_trace(go.Scatter(x=panel_df.index, y=panel_df["%D"],
                line=dict(color="orange", width=1), name="%D"), row=current_row, col=1)
            fig.update_yaxes(range=[0, 100], row=current_row, col=1)

        elif panel_name == "obv":
            fig.add_trace(go.Scatter(x=panel_df.index, y=panel_df["OBV"],
                line=dict(color="purple", width=1), name="OBV"), row=current_row, col=1)

        current_row += 1

    fig.update_layout(
        height=200 * (1 + n_panels),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        template="plotly_dark",
        margin=dict(l=40, r=40, t=20, b=40),
    )

    return fig
