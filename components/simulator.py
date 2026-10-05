import numpy as np
import pandas as pd
import plotly.graph_objects as go
from components.charts import apply_chart_style

def simulate_investment(forecast_df, current_price, initial_capital=1000.0, stop_loss_pct=5.0, take_profit_pct=15.0):
    """
    Simulate portfolio growth, return on investment (ROI), and risk scenarios
    based on the AI forecasted price trajectory.
    """
    if forecast_df is None or forecast_df.empty or current_price <= 0:
        return {}, None

    # Number of coins purchased with initial capital
    coins_held = initial_capital / current_price

    # Trajectory values
    dates = forecast_df["Date"]
    predicted_prices = forecast_df["Predicted_Price"].values
    upper_prices = forecast_df["Upper_Bound"].values if "Upper_Bound" in forecast_df.columns else predicted_prices
    lower_prices = forecast_df["Lower_Bound"].values if "Lower_Bound" in forecast_df.columns else predicted_prices

    portfolio_values = coins_held * predicted_prices
    upper_values = coins_held * upper_prices
    lower_values = coins_held * lower_prices

    final_val = portfolio_values[-1]
    net_profit = final_val - initial_capital
    roi_pct = (net_profit / initial_capital) * 100

    best_val = upper_values[-1]
    best_profit = best_val - initial_capital
    best_roi = (best_profit / initial_capital) * 100

    worst_val = lower_values[-1]
    worst_profit = worst_val - initial_capital
    worst_roi = (worst_profit / initial_capital) * 100

    max_drawdown = float(np.min(portfolio_values - initial_capital))
    max_drawdown_pct = (max_drawdown / initial_capital) * 100 if max_drawdown < 0 else 0.0

    summary = {
        "initial_capital": initial_capital,
        "coins_held": coins_held,
        "final_value": final_val,
        "net_profit": net_profit,
        "roi_pct": roi_pct,
        "best_value": best_val,
        "best_roi": best_roi,
        "worst_value": worst_val,
        "worst_roi": worst_roi,
        "max_drawdown_pct": max_drawdown_pct,
    }

    # Generate Equity Curve Chart
    fig = go.Figure()

    # Upper Scenario
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=upper_values,
            mode="lines",
            name="Bullish Scenario (Upper Bound)",
            line=dict(color="rgba(16, 185, 129, 0.4)", width=1, dash="dot"),
        )
    )

    # Lower Scenario
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=lower_values,
            mode="lines",
            name="Bearish Scenario (Lower Bound)",
            line=dict(color="rgba(239, 68, 68, 0.4)", width=1, dash="dot"),
            fill="tonexty",
            fillcolor="rgba(0, 242, 254, 0.05)",
        )
    )

    # Base Expected Equity
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=portfolio_values,
            mode="lines+markers",
            name="Expected Portfolio Value",
            line=dict(color="#00F2FE", width=3),
            marker=dict(size=6, color="#00F2FE"),
            hovertemplate="<b>Date:</b> %{x|%b %d}<br><b>Portfolio Value:</b> $%{y:,.2f}<extra></extra>",
        )
    )

    # Initial Capital baseline
    fig.add_hline(
        y=initial_capital,
        line_dash="dash",
        line_color="rgba(255, 255, 255, 0.4)",
        annotation_text="Initial Capital ($1,000)",
        annotation_position="bottom left",
    )

    fig.update_layout(
        title="<b>Simulated Portfolio Equity Curve ($)</b>",
        height=400,
    )
    apply_chart_style(fig)

    return summary, fig
