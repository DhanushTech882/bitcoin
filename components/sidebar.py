import streamlit as st

def sidebar_controls():
    st.sidebar.title("⚙ Forecast Settings")

    lookback = st.sidebar.slider("Lookback Days", 30, 120, 60)
    future_days = st.sidebar.slider("Forecast Days", 7, 30, 10)

    return lookback, future_days
