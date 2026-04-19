import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
from tensorflow.keras.models import load_model

from components.sidebar import sidebar_controls
from components.forecast import generate_forecast

st.set_page_config(page_title="Bitcoin AI Dashboard", layout="wide")

# Load CSS
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("<h1>₿ Bitcoin AI Dashboard</h1>", unsafe_allow_html=True)

if not os.path.exists("data/crypto_data.csv"):
    st.error("Run train_model.py first")
    st.stop()

df = pd.read_csv("data/crypto_data.csv")
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
df.dropna(inplace=True)

close_prices = df[["Close"]].values
model = load_model("models/lstm_crypto_model.keras", compile=False)

lookback, future_days = sidebar_controls()

latest_price = close_prices[-1][0]

col1, col2 = st.columns(2)
col1.metric("Current BTC Price", f"${latest_price:,.2f}")
col2.metric("Total Records", len(df))

if st.button("🚀 Generate Forecast"):

    future_prices = generate_forecast(
        model,
        close_prices,
        lookback,
        future_days
    )

    forecast_df = pd.DataFrame({
        "Day": [f"Day {i+1}" for i in range(future_days)],
        "Predicted Price": future_prices.flatten()
    })

    # Historical + Forecast Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=df["Close"],
        mode="lines",
        name="Historical Price"
    ))

    fig.add_trace(go.Scatter(
        y=forecast_df["Predicted Price"],
        mode="lines+markers",
        name="Forecast"
    ))

    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # Profit chart
    profit = forecast_df["Predicted Price"] - latest_price

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(y=profit))
    fig2.update_layout(
        title="Expected Profit/Loss",
        template="plotly_dark"
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(forecast_df)

    csv = forecast_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download Forecast Report",
        csv,
        "bitcoin_forecast.csv",
        "text/csv"
    )

    st.success("Forecast Generated Successfully!")
