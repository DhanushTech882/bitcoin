import os
import pandas as pd
import streamlit as st
import yfinance as yf

# Mapping common crypto names to yfinance tickers
CRYPTO_TICKERS = {
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "Solana (SOL)": "SOL-USD",
    "Binance Coin (BNB)": "BNB-USD",
    "Cardano (ADA)": "ADA-USD",
    "Ripple (XRP)": "XRP-USD",
    "Dogecoin (DOGE)": "DOGE-USD",
    "Avalanche (AVAX)": "AVAX-USD",
}

@st.cache_data(ttl=300, show_spinner=False)
def fetch_crypto_data(ticker="BTC-USD", period="1y"):
    """
    Fetch historical and live crypto data using yfinance with fallback to local cache.
    """
    try:
        df = yf.download(ticker, period=period, interval="1d", progress=False)
        if df is not None and not df.empty and len(df) > 5:
            # Flatten multi-index columns if returned by yfinance
            if hasattr(df.columns, 'nlevels') and df.columns.nlevels > 1:
                df.columns = df.columns.get_level_values(0)
            elif isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Ensure numeric columns
            numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            df.dropna(subset=["Close"], inplace=True)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)

            # Save BTC data to local cache file for offline resilience
            if ticker == "BTC-USD":
                os.makedirs("data", exist_ok=True)
                df.to_csv("data/crypto_data.csv")

            return df, "live"
    except Exception as e:
        print(f"Live fetch error for {ticker}: {e}")

    # Fallback to local cache if offline or fetch failed
    local_path = "data/crypto_data.csv"
    if os.path.exists(local_path):
        try:
            df = pd.read_csv(local_path, index_col=0)
            if hasattr(df.columns, 'nlevels') and df.columns.nlevels > 1:
                df.columns = df.columns.get_level_values(0)
            elif isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if "Close" in df.columns:
                df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
                df.dropna(subset=["Close"], inplace=True)
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                df.sort_index(inplace=True)
                return df, "cached"
        except Exception as e:
            print(f"Local cache read error: {e}")

    return pd.DataFrame(), "error"


def calculate_market_kpis(df):
    """
    Calculate essential market metrics:
    Current Price, 24h Change, 24h High/Low, 24h Volume, 7d Volatility.
    """
    if df is None or len(df) < 2:
        return {}

    current_price = float(df["Close"].iloc[-1])
    prev_price = float(df["Close"].iloc[-2])
    change_24h = current_price - prev_price
    pct_change_24h = (change_24h / prev_price) * 100

    high_24h = float(df["High"].iloc[-1]) if "High" in df.columns else current_price
    low_24h = float(df["Low"].iloc[-1]) if "Low" in df.columns else current_price
    volume_24h = float(df["Volume"].iloc[-1]) if "Volume" in df.columns else 0.0

    # 7-day annualized volatility
    returns = df["Close"].pct_change().dropna()
    volatility_7d = float(returns.tail(7).std() * (365 ** 0.5) * 100) if len(returns) >= 7 else 0.0

    # 52-week or all-time high in dataset
    ath_period = float(df["Close"].max())

    return {
        "current_price": current_price,
        "prev_price": prev_price,
        "change_24h": change_24h,
        "pct_change_24h": pct_change_24h,
        "high_24h": high_24h,
        "low_24h": low_24h,
        "volume_24h": volume_24h,
        "volatility_7d": volatility_7d,
        "ath_period": ath_period,
    }
