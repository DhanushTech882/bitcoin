import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Premium theme palette
THEME_BG = "#0B0F19"
THEME_CARD_BG = "#121826"
GRID_COLOR = "rgba(255, 255, 255, 0.06)"
FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif"

def apply_chart_style(fig):
    """Apply consistent high-end styling to any Plotly figure."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(11, 15, 25, 0.7)",
        font=dict(family=FONT_FAMILY, color="#94A3B8"),
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color="#CBD5E1"),
            bgcolor="rgba(18, 24, 38, 0.6)",
        ),
        xaxis=dict(
            gridcolor=GRID_COLOR,
            zerolinecolor=GRID_COLOR,
            showline=True,
            linecolor="rgba(255,255,255,0.1)",
        ),
        yaxis=dict(
            gridcolor=GRID_COLOR,
            zerolinecolor=GRID_COLOR,
            showline=True,
            linecolor="rgba(255,255,255,0.1)",
        ),
    )
    return fig


def create_candlestick_chart(df, ticker="BTC-USD", show_indicators=None):
    """
    Create an interactive Candlestick chart with Volume bars and selectable overlays.
    """
    if show_indicators is None:
        show_indicators = ["SMA", "Bollinger"]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.75, 0.25],
        subplot_titles=(f"<b>{ticker}</b> Price Action", "Volume (USD)")
    )

    # Candlestick Trace
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"] if "Open" in df.columns else df["Close"],
            high=df["High"] if "High" in df.columns else df["Close"],
            low=df["Low"] if "Low" in df.columns else df["Close"],
            close=df["Close"],
            name="OHLC",
            increasing_line_color="#10B981",
            decreasing_line_color="#EF4444",
            increasing_fillcolor="#10B981",
            decreasing_fillcolor="#EF4444",
        ),
        row=1, col=1
    )

    # Indicator Overlays
    if "Bollinger" in show_indicators and "BB_Upper" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["BB_Upper"],
                name="BB Upper",
                line=dict(color="rgba(168, 85, 247, 0.6)", width=1, dash="dot"),
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["BB_Lower"],
                name="BB Lower",
                line=dict(color="rgba(168, 85, 247, 0.6)", width=1, dash="dot"),
                fill="tonexty",
                fillcolor="rgba(168, 85, 247, 0.06)",
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["BB_Middle"],
                name="BB Middle (SMA 20)",
                line=dict(color="rgba(168, 85, 247, 0.8)", width=1),
            ),
            row=1, col=1
        )

    if "SMA" in show_indicators:
        if "SMA_20" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df["SMA_20"],
                    name="SMA 20",
                    line=dict(color="#38BDF8", width=1.5),
                ),
                row=1, col=1
            )
        if "SMA_50" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df["SMA_50"],
                    name="SMA 50",
                    line=dict(color="#F59E0B", width=1.5),
                ),
                row=1, col=1
            )

    if "EMA" in show_indicators and "EMA_20" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["EMA_20"],
                name="EMA 20",
                line=dict(color="#EC4899", width=1.5, dash="dash"),
            ),
            row=1, col=1
        )

    # Volume Subplot
    if "Volume" in df.columns:
        vol_colors = [
            "#10B981" if (c >= o) else "#EF4444"
            for c, o in zip(df["Close"], df["Open"] if "Open" in df.columns else df["Close"])
        ]
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["Volume"],
                name="Volume",
                marker_color=vol_colors,
                opacity=0.65,
            ),
            row=2, col=1
        )

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=580,
    )
    apply_chart_style(fig)
    return fig


def create_rsi_chart(df):
    """Create interactive RSI 14 oscillator chart with 70/30 thresholds."""
    if "RSI_14" not in df.columns:
        return go.Figure()

    fig = go.Figure()

    # RSI Line
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["RSI_14"],
            name="RSI 14",
            line=dict(color="#00F2FE", width=2),
        )
    )

    # Overbought Threshold (70)
    fig.add_hline(
        y=70, line_dash="dash", line_color="#EF4444", line_width=1.5,
        annotation_text="Overbought (70)", annotation_position="top right",
        annotation_font_color="#EF4444"
    )

    # Neutral Line (50)
    fig.add_hline(
        y=50, line_dash="dot", line_color="rgba(255,255,255,0.2)", line_width=1
    )

    # Oversold Threshold (30)
    fig.add_hline(
        y=30, line_dash="dash", line_color="#10B981", line_width=1.5,
        annotation_text="Oversold (30)", annotation_position="bottom right",
        annotation_font_color="#10B981"
    )

    fig.update_layout(
        title="<b>Relative Strength Index (RSI 14)</b>",
        yaxis=dict(range=[10, 90]),
        height=280,
    )
    apply_chart_style(fig)
    return fig


def create_macd_chart(df):
    """Create interactive MACD chart with signal line and histogram."""
    if "MACD" not in df.columns:
        return go.Figure()

    fig = go.Figure()

    # Histogram
    hist_colors = ["#10B981" if val >= 0 else "#EF4444" for val in df["MACD_Hist"]]
    fig.add_trace(
        go.Bar(
            x=df.index, y=df["MACD_Hist"],
            name="Histogram",
            marker_color=hist_colors,
            opacity=0.6,
        )
    )

    # MACD Line
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["MACD"],
            name="MACD Line (12, 26)",
            line=dict(color="#38BDF8", width=2),
        )
    )

    # Signal Line
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["MACD_Signal"],
            name="Signal Line (9)",
            line=dict(color="#F59E0B", width=1.5, dash="dash"),
        )
    )

    fig.update_layout(
        title="<b>MACD (Moving Average Convergence Divergence)</b>",
        height=300,
    )
    apply_chart_style(fig)
    return fig


def create_forecast_chart(historical_df, forecast_df, ticker="BTC"):
    """
    Combined chart of historical price and future LSTM predictions with confidence bounds.
    """
    fig = go.Figure()

    # Use recent historical window for clear zoom (e.g. last 90 days)
    recent_hist = historical_df.tail(90)

    # 1. Historical Prices
    fig.add_trace(
        go.Scatter(
            x=recent_hist.index,
            y=recent_hist["Close"],
            mode="lines",
            name="Historical Price",
            line=dict(color="#94A3B8", width=2),
            hovertemplate="<b>Date:</b> %{x|%b %d, %Y}<br><b>Close:</b> $%{y:,.2f}<extra></extra>",
        )
    )

    # 2. Confidence Band (Upper and Lower bounds)
    if "Upper_Bound" in forecast_df.columns and "Lower_Bound" in forecast_df.columns:
        fig.add_trace(
            go.Scatter(
                x=forecast_df["Date"],
                y=forecast_df["Upper_Bound"],
                mode="lines",
                name="Upper Bound (95% CI)",
                line=dict(color="rgba(0, 242, 254, 0.4)", width=1, dash="dot"),
                showlegend=True,
                hovertemplate="<b>Date:</b> %{x|%b %d, %Y}<br><b>Upper:</b> $%{y:,.2f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_df["Date"],
                y=forecast_df["Lower_Bound"],
                mode="lines",
                name="Lower Bound (95% CI)",
                line=dict(color="rgba(0, 242, 254, 0.4)", width=1, dash="dot"),
                fill="tonexty",
                fillcolor="rgba(0, 242, 254, 0.10)",
                showlegend=True,
                hovertemplate="<b>Date:</b> %{x|%b %d, %Y}<br><b>Lower:</b> $%{y:,.2f}<extra></extra>",
            )
        )

    # 3. AI Forecast Line
    fig.add_trace(
        go.Scatter(
            x=forecast_df["Date"],
            y=forecast_df["Predicted_Price"],
            mode="lines+markers",
            name="AI Forecast",
            line=dict(color="#00F2FE", width=3),
            marker=dict(size=6, color="#00F2FE", symbol="circle"),
            hovertemplate="<b>Date:</b> %{x|%b %d, %Y}<br><b>Forecast:</b> $%{y:,.2f}<extra></extra>",
        )
    )

    # Connect historical last point to first prediction
    last_hist_date = recent_hist.index[-1]
    last_hist_price = recent_hist["Close"].iloc[-1]
    first_pred_date = forecast_df["Date"].iloc[0]
    first_pred_price = forecast_df["Predicted_Price"].iloc[0]

    fig.add_trace(
        go.Scatter(
            x=[last_hist_date, first_pred_date],
            y=[last_hist_price, first_pred_price],
            mode="lines",
            line=dict(color="#00F2FE", width=2, dash="dot"),
            showlegend=False,
            hoverinfo="skip"
        )
    )

    # Boundary Line marking start of forecast
    fig.add_shape(
        type="line",
        x0=last_hist_date,
        x1=last_hist_date,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="#F7931A", width=1.5, dash="dash")
    )
    fig.add_annotation(
        x=last_hist_date,
        y=1.02,
        yref="paper",
        text="Forecast Start",
        showarrow=False,
        xanchor="right",
        font=dict(color="#F7931A", size=11)
    )

    fig.update_layout(
        title=f"<b>{ticker} AI Price Forecast with Confidence Interval</b>",
        height=520,
    )
    apply_chart_style(fig)
    return fig


def create_profit_loss_chart(forecast_df, current_price):
    """Bar chart of projected net profit/loss relative to current price."""
    diff = forecast_df["Predicted_Price"] - current_price
    pct_diff = (diff / current_price) * 100

    bar_colors = ["#10B981" if val >= 0 else "#EF4444" for val in diff]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=forecast_df["Date"].dt.strftime("%b %d"),
            y=diff,
            name="Profit / Loss ($)",
            marker_color=bar_colors,
            customdata=pct_diff,
            hovertemplate="<b>Date:</b> %{x}<br><b>Expected P/L:</b> $%{y:,.2f}<br><b>ROI:</b> %{customdata:+.2f}%<extra></extra>",
        )
    )

    fig.add_hline(y=0, line_color="rgba(255,255,255,0.3)", line_width=1)

    fig.update_layout(
        title="<b>Expected Profit / Loss per Coin ($)</b>",
        height=320,
    )
    apply_chart_style(fig)
    return fig
