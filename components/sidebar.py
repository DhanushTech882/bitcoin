import streamlit as st
from components.data_loader import CRYPTO_TICKERS

def sidebar_controls():
    """
    Renders sidebar controls and returns user configuration dictionary.
    """
    st.sidebar.markdown(
        """
        <div style="text-align: center; padding-bottom: 15px;">
            <h2 style="margin: 0; color: #F7931A;">₿ Crypto AI Terminal</h2>
            <p style="margin: 0; font-size: 0.82rem; color: #94A3B8;">Real-Time Deep Learning Analytics</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Asset Selection")

    crypto_options = list(CRYPTO_TICKERS.keys()) + ["Custom Ticker..."]
    selected_asset = st.sidebar.selectbox("Select Cryptocurrency", crypto_options, index=0)

    if selected_asset == "Custom Ticker...":
        ticker = st.sidebar.text_input("Enter Yahoo Finance Ticker (e.g. DOGE-USD)", value="DOGE-USD").strip().upper()
    else:
        ticker = CRYPTO_TICKERS[selected_asset]

    # Historical Timeframe
    period_options = {
        "1 Month": "1mo",
        "3 Months": "3mo",
        "6 Months": "6mo",
        "1 Year": "1y",
        "2 Years": "2y",
        "5 Years": "5y",
        "Max Historical": "max",
    }
    selected_period_label = st.sidebar.selectbox("Historical Timeframe", list(period_options.keys()), index=3)
    period = period_options[selected_period_label]

    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 AI Forecast Settings")

    future_days = st.sidebar.slider("Forecast Horizon (Days)", min_value=3, max_value=60, value=14, step=1)
    lookback = st.sidebar.slider("Model Lookback Window (Days)", min_value=20, max_value=120, value=60, step=5)

    confidence_label = st.sidebar.select_slider(
        "Confidence Interval",
        options=["80%", "90%", "95%"],
        value="95%"
    )
    confidence_level = float(confidence_label.replace("%", "")) / 100.0

    st.sidebar.markdown("---")
    st.sidebar.subheader("Chart Overlays")
    show_bb = st.sidebar.checkbox("Bollinger Bands (20, 2σ)", value=True)
    show_sma = st.sidebar.checkbox("Moving Averages (SMA 20/50)", value=True)
    show_ema = st.sidebar.checkbox("Exponential MA (EMA 20)", value=False)

    overlays = []
    if show_bb:
        overlays.append("Bollinger")
    if show_sma:
        overlays.append("SMA")
    if show_ema:
        overlays.append("EMA")

    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Sync Live Market Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    return {
        "asset_name": selected_asset,
        "ticker": ticker,
        "period": period,
        "future_days": future_days,
        "lookback": lookback,
        "confidence_level": confidence_level,
        "overlays": overlays,
    }
