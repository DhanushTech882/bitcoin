import os
import streamlit as st
import pandas as pd
import numpy as np

from components.sidebar import sidebar_controls
from components.data_loader import fetch_crypto_data, calculate_market_kpis
from components.indicators import add_technical_indicators, generate_technical_signals
from components.charts import (
    create_candlestick_chart,
    create_rsi_chart,
    create_macd_chart,
    create_forecast_chart,
    create_profit_loss_chart,
)
from components.forecast import (
    get_or_load_model,
    generate_forecast,
    evaluate_model_metrics,
)
from components.simulator import simulate_investment

# Page Configuration
st.set_page_config(
    page_title="Crypto AI Analytics Terminal",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Custom Cyber / FinTech CSS
css_file = "assets/style.css"
if os.path.exists(css_file):
    with open(css_file, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Render Sidebar Controls
config = sidebar_controls()
ticker = config["ticker"]
asset_name = config["asset_name"]

# Ingest Crypto Data (Live via yfinance with cached fallback)
with st.spinner(f"Loading market data for {ticker}..."):
    df, status = fetch_crypto_data(ticker=ticker, period=config["period"])

if df is None or df.empty or len(df) < 5:
    st.error(f"⚠️ Unable to load market data for ticker '{ticker}'. Please verify the ticker symbol or check your network connection.")
    st.stop()

# Compute KPIs and Technical Indicators
kpis = calculate_market_kpis(df)
df_indicators = add_technical_indicators(df)
overall_signal, signal_score, signal_reasons = generate_technical_signals(df_indicators)

# Load AI Model
model = get_or_load_model()

# Header Section
status_badge = (
    '<span class="live-badge"><span class="pulse-dot"></span> LIVE FEED</span>'
    if status == "live"
    else '<span class="live-badge" style="color: #F59E0B; border-color: rgba(245,158,11,0.3); background: rgba(245,158,11,0.1);">⚡ CACHED DATA</span>'
)

st.markdown(
    f"""
    <div class="header-container">
        <div class="header-title">
            <h1>₿ {asset_name} AI Terminal</h1>
        </div>
        <div>
            {status_badge}
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Top KPI Metric Row
if kpis:
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # 24h Change Delta formatting
    delta_arrow = "▲" if kpis["change_24h"] >= 0 else "▼"
    delta_class = "metric-delta-pos" if kpis["change_24h"] >= 0 else "metric-delta-neg"
    
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Current Price</div>
                <div class="metric-value">${kpis['current_price']:,.2f}</div>
                <div class="{delta_class}">{delta_arrow} ${abs(kpis['change_24h']):,.2f} ({kpis['pct_change_24h']:+.2f}%)</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">24h High / Low</div>
                <div class="metric-value" style="font-size: 1.15rem;">${kpis['high_24h']:,.2f}</div>
                <div style="font-size: 0.8rem; color: #94A3B8;">Low: ${kpis['low_24h']:,.2f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        vol_str = f"${kpis['volume_24h']/1e9:.2f}B" if kpis['volume_24h'] >= 1e9 else f"${kpis['volume_24h']/1e6:.2f}M"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">24h Volume</div>
                <div class="metric-value">{vol_str}</div>
                <div style="font-size: 0.8rem; color: #94A3B8;">Total 24h Trading Vol</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">7-Day Volatility</div>
                <div class="metric-value">{kpis['volatility_7d']:.1f}%</div>
                <div style="font-size: 0.8rem; color: #94A3B8;">Annualized Risk Index</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col5:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Period High</div>
                <div class="metric-value">${kpis['ath_period']:,.2f}</div>
                <div style="font-size: 0.8rem; color: #94A3B8;">Selected Timeframe Peak</div>
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

# Main Navigation Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Market Overview",
    "📈 Technical Analysis",
    "🤖 AI Forecast",
    "💰 Investment Simulator",
    "⚙️ Model Diagnostics"
])

# -------------------------------------------------------------
# TAB 1: MARKET OVERVIEW
# -------------------------------------------------------------
with tab1:
    st.markdown("### Interactive Candlestick & Volume Chart")
    fig_candle = create_candlestick_chart(
        df_indicators,
        ticker=ticker,
        show_indicators=config["overlays"]
    )
    st.plotly_chart(fig_candle, use_container_width=True)

    with st.expander("🔍 View Raw Historical Price Data"):
        st.dataframe(
            df_indicators[["Open", "High", "Low", "Close", "Volume"]].tail(50).sort_index(ascending=False),
            use_container_width=True
        )

# -------------------------------------------------------------
# TAB 2: TECHNICAL ANALYSIS & SIGNALS
# -------------------------------------------------------------
with tab2:
    st.markdown("### Technical Momentum & Automated Signals")

    badge_class_map = {
        "STRONG BUY": "signal-badge-strong-buy",
        "BUY": "signal-badge-buy",
        "NEUTRAL": "signal-badge-neutral",
        "SELL": "signal-badge-sell",
        "STRONG SELL": "signal-badge-strong-sell",
    }
    badge_cls = badge_class_map.get(overall_signal, "signal-badge-neutral")

    sig_col1, sig_col2 = st.columns([1, 2])
    with sig_col1:
        st.markdown(
            f"""
            <div class="signal-box" style="text-align: center;">
                <div style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 8px;">COMPOSITE SIGNAL</div>
                <div class="{badge_cls}">{overall_signal}</div>
                <div style="margin-top: 12px; font-size: 0.9rem; color: #94A3B8;">
                    Momentum Score: <b style="color: #FFFFFF;">{signal_score:+d} / 100</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with sig_col2:
        st.markdown(
            """
            <div class="signal-box">
                <div style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 8px;">ACTIVE INDICATOR SIGNALS</div>
            """,
            unsafe_allow_html=True
        )
        for reason in signal_reasons:
            st.markdown(f"- {reason}")
        st.markdown("</div>", unsafe_allow_html=True)

    # Secondary Oscillators
    t_col1, t_col2 = st.columns(2)
    with t_col1:
        fig_rsi = create_rsi_chart(df_indicators)
        st.plotly_chart(fig_rsi, use_container_width=True)
    with t_col2:
        fig_macd = create_macd_chart(df_indicators)
        st.plotly_chart(fig_macd, use_container_width=True)

# -------------------------------------------------------------
# TAB 3: AI FORECAST ENGINE
# -------------------------------------------------------------
with tab3:
    st.markdown(f"### LSTM Neural Network Price Projections ({config['future_days']} Days Ahead)")

    close_data = df[["Close"]].values
    last_historical_date = df.index[-1]

    # Pre-generate forecast or generate on button
    forecast_df = generate_forecast(
        model=model,
        close_prices=close_data,
        lookback=config["lookback"],
        future_days=config["future_days"],
        last_date=last_historical_date,
        confidence_level=config["confidence_level"]
    )

    # Forecast Main Chart with Confidence Band
    fig_forecast = create_forecast_chart(df, forecast_df, ticker=ticker)
    st.plotly_chart(fig_forecast, use_container_width=True)

    # Forecast KPI cards
    target_day_pred = forecast_df["Predicted_Price"].iloc[-1]
    target_day_roi = forecast_df["Expected_ROI_Pct"].iloc[-1]
    target_day_diff = forecast_df["Expected_Profit"].iloc[-1]
    upper_target = forecast_df["Upper_Bound"].iloc[-1]
    lower_target = forecast_df["Lower_Bound"].iloc[-1]

    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    with f_col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Target Day Forecast</div>
                <div class="metric-value">${target_day_pred:,.2f}</div>
                <div style="font-size: 0.8rem; color: #94A3B8;">Day {config['future_days']} Projected Price</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with f_col2:
        f_delta_cls = "metric-delta-pos" if target_day_diff >= 0 else "metric-delta-neg"
        f_delta_arrow = "▲" if target_day_diff >= 0 else "▼"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Projected Net ROI</div>
                <div class="metric-value">{target_day_roi:+.2f}%</div>
                <div class="{f_delta_cls}">{f_delta_arrow} ${abs(target_day_diff):,.2f} per coin</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with f_col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Bullish Upper Bound</div>
                <div class="metric-value" style="color: #10B981;">${upper_target:,.2f}</div>
                <div style="font-size: 0.8rem; color: #94A3B8;">{int(config['confidence_level']*100)}% Confidence Ceiling</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with f_col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Bearish Lower Bound</div>
                <div class="metric-value" style="color: #EF4444;">${lower_target:,.2f}</div>
                <div style="font-size: 0.8rem; color: #94A3B8;">{int(config['confidence_level']*100)}% Confidence Floor</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # Profit / Loss Bar Chart
    fig_pl = create_profit_loss_chart(forecast_df, current_price=kpis["current_price"])
    st.plotly_chart(fig_pl, use_container_width=True)

    # Forecast Breakdown Table & Download
    st.markdown("#### Day-by-Day Forecast Breakdown")
    
    display_forecast = forecast_df.copy()
    display_forecast["Date"] = display_forecast["Date"].dt.strftime("%Y-%m-%d")
    
    st.dataframe(
        display_forecast[["Date", "Day", "Predicted_Price", "Upper_Bound", "Lower_Bound", "Expected_Profit", "Expected_ROI_Pct"]],
        use_container_width=True
    )

    csv_data = display_forecast.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Detailed Forecast (CSV)",
        data=csv_data,
        file_name=f"{ticker}_ai_forecast.csv",
        mime="text/csv",
        use_container_width=True
    )

# -------------------------------------------------------------
# TAB 4: INVESTMENT & ROI SIMULATOR
# -------------------------------------------------------------
with tab4:
    st.markdown("### Interactive Portfolio & Investment Simulator")
    st.write("Simulate expected returns and risk scenarios for your portfolio based on the AI price projection.")

    sim_col1, sim_col2 = st.columns([1, 2])
    with sim_col1:
        st.markdown("#### Investment Parameters")
        capital_input = st.number_input(
            "Initial Investment Capital ($ USD)",
            min_value=10.0,
            max_value=10000000.0,
            value=1000.0,
            step=100.0
        )

        sim_summary, fig_sim = simulate_investment(
            forecast_df=forecast_df,
            current_price=kpis["current_price"],
            initial_capital=capital_input
        )

        if sim_summary:
            st.markdown(
                f"""
                <div style="background: rgba(14, 19, 31, 0.8); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 16px; margin-top: 15px;">
                    <div style="font-size: 0.85rem; color: #94A3B8;">Asset Quantity Acquired:</div>
                    <div style="font-size: 1.1rem; font-weight: bold; color: #FFFFFF; font-family: monospace;">
                        {sim_summary['coins_held']:.6f} {ticker.split('-')[0]}
                    </div>
                    <hr style="border: 0; border-top: 1px solid rgba(255,255,255,0.08); margin: 10px 0;">
                    <div style="font-size: 0.85rem; color: #94A3B8;">Projected Final Value:</div>
                    <div style="font-size: 1.35rem; font-weight: bold; color: #00F2FE; font-family: monospace;">
                        ${sim_summary['final_value']:,.2f}
                    </div>
                    <div style="font-size: 0.85rem; color: {'#10B981' if sim_summary['net_profit'] >= 0 else '#EF4444'}; margin-top: 4px;">
                        {'▲' if sim_summary['net_profit'] >= 0 else '▼'} ${abs(sim_summary['net_profit']):,.2f} ({sim_summary['roi_pct']:+.2f}%)
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    with sim_col2:
        if fig_sim:
            st.plotly_chart(fig_sim, use_container_width=True)

        if sim_summary:
            sc_col1, sc_col2 = st.columns(2)
            with sc_col1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Best-Case Outcome (Upper)</div>
                        <div class="metric-value" style="color: #10B981;">${sim_summary['best_value']:,.2f}</div>
                        <div class="metric-delta-pos">ROI: {sim_summary['best_roi']:+.2f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with sc_col2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Worst-Case Outcome (Lower)</div>
                        <div class="metric-value" style="color: #EF4444;">${sim_summary['worst_value']:,.2f}</div>
                        <div class="metric-delta-neg">ROI: {sim_summary['worst_roi']:+.2f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# -------------------------------------------------------------
# TAB 5: MODEL DIAGNOSTICS & RETRAINING
# -------------------------------------------------------------
with tab5:
    st.markdown("### Deep Learning Architecture & Accuracy Metrics")

    metrics = evaluate_model_metrics(model, close_data, lookback=config["lookback"])

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">RMSE (Root Mean Sq Error)</div>
                <div class="metric-value">${metrics['rmse']:,.2f}</div>
                <div style="font-size: 0.8rem; color: #94A3B8;">On Last {metrics['sample_size']} Test Samples</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with m_col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">MAE (Mean Absolute Error)</div>
                <div class="metric-value">${metrics['mae']:,.2f}</div>
                <div style="font-size: 0.8rem; color: #94A3B8;">Average Dollar Deviation</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with m_col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">MAPE (Pct Error)</div>
                <div class="metric-value">{metrics['mape']:.2f}%</div>
                <div style="font-size: 0.8rem; color: #94A3B8;">Relative Deviation</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with m_col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Directional Accuracy</div>
                <div class="metric-value" style="color: #10B981;">{metrics['directional_accuracy']:.1f}%</div>
                <div style="font-size: 0.8rem; color: #94A3B8;">Trend Hit Rate</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("#### Neural Network Specifications")

    spec_col1, spec_col2 = st.columns(2)
    with spec_col1:
        st.markdown(
            """
            - **Model Type**: Deep Recurrent Neural Network (Stacked LSTM)
            - **Layers**:
                - `LSTM(64 units, return_sequences=True)`
                - `Dropout(0.20)`
                - `LSTM(64 units)`
                - `Dropout(0.20)`
                - `Dense(32 units, activation='relu')`
                - `Dense(1 unit, linear output)`
            - **Optimizer**: Adam (learning rate = 0.001)
            - **Loss Function**: Mean Squared Error (MSE)
            """
        )

    with spec_col2:
        training_loss_path = "outputs/training_loss.png"
        if os.path.exists(training_loss_path):
            st.image(training_loss_path, caption="LSTM Training vs Validation Loss Curve", use_container_width=True)
        else:
            st.info("Loss plot will appear once a training run completes.")

    st.markdown("---")
    st.markdown("#### In-App Model Retraining")
    st.write("Retrain the neural network on the most recent live data from Yahoo Finance.")

    if st.button("⚡ Retrain Neural Network Now", use_container_width=True):
        with st.status("Training LSTM model on latest market data...", expanded=True) as status_box:
            st.write("📥 Fetching latest Bitcoin market history...")
            from sklearn.preprocessing import MinMaxScaler
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, LSTM, Dropout
            import matplotlib.pyplot as plt

            # Download freshest training data
            train_df, _ = fetch_crypto_data(ticker="BTC-USD", period="5y")
            data_arr = train_df[["Close"]].dropna().values

            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data_arr)

            lb = 60
            X, y = [], []
            for i in range(lb, len(scaled_data)):
                X.append(scaled_data[i-lb:i, 0])
                y.append(scaled_data[i, 0])

            X, y = np.array(X), np.array(y)
            X = X.reshape(X.shape[0], X.shape[1], 1)

            split = int(0.85 * len(X))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            st.write(f"🧠 Training model across {len(X_train)} samples...")
            new_model = Sequential([
                LSTM(64, return_sequences=True, input_shape=(lb, 1)),
                Dropout(0.2),
                LSTM(64),
                Dropout(0.2),
                Dense(32),
                Dense(1)
            ])
            new_model.compile(optimizer="adam", loss="mean_squared_error")

            history = new_model.fit(
                X_train, y_train,
                epochs=8,
                batch_size=32,
                validation_data=(X_test, y_test),
                verbose=0
            )

            os.makedirs("models", exist_ok=True)
            new_model.save("models/lstm_crypto_model.keras")

            # Save loss plot
            os.makedirs("outputs", exist_ok=True)
            plt.figure(figsize=(10, 4))
            plt.plot(history.history["loss"], label="Train Loss", color="#00F2FE")
            plt.plot(history.history["val_loss"], label="Validation Loss", color="#F7931A")
            plt.legend()
            plt.grid(True, alpha=0.2)
            plt.savefig("outputs/training_loss.png", bbox_inches="tight")
            plt.close()

            status_box.update(label="✅ Model Retrained & Saved Successfully!", state="complete", expanded=False)
            st.success("New model checkpoint saved to `models/lstm_crypto_model.keras`! Reloading app...")
            st.rerun()
