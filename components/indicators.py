import numpy as np
import pandas as pd

def add_technical_indicators(df):
    """
    Calculate and append popular technical indicators to the DataFrame:
    SMA (20, 50), EMA (20), Bollinger Bands (Upper, Middle, Lower),
    RSI (14), MACD, and Signal Line.
    """
    data = df.copy()
    close = data["Close"]

    # Simple Moving Averages
    data["SMA_20"] = close.rolling(window=20).mean()
    data["SMA_50"] = close.rolling(window=50).mean()

    # Exponential Moving Averages
    data["EMA_20"] = close.ewm(span=20, adjust=False).mean()
    data["EMA_50"] = close.ewm(span=50, adjust=False).mean()

    # Bollinger Bands (20-day, 2 std dev)
    rolling_std = close.rolling(window=20).std()
    data["BB_Middle"] = data["SMA_20"]
    data["BB_Upper"] = data["BB_Middle"] + (rolling_std * 2)
    data["BB_Lower"] = data["BB_Middle"] - (rolling_std * 2)
    data["BB_Width"] = (data["BB_Upper"] - data["BB_Lower"]) / data["BB_Middle"]

    # Relative Strength Index (RSI 14)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    data["RSI_14"] = 100 - (100 / (1 + rs))

    # MACD (12-period EMA - 26-period EMA) & Signal line (9-period EMA)
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    data["MACD"] = ema_12 - ema_26
    data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
    data["MACD_Hist"] = data["MACD"] - data["MACD_Signal"]

    return data


def generate_technical_signals(df):
    """
    Evaluate technical indicator readings to produce an AI composite trading signal.
    Returns:
        signal: "STRONG BUY" | "BUY" | "NEUTRAL" | "SELL" | "STRONG SELL"
        score: int (-100 to +100)
        breakdown: list of signal explanations
    """
    if df is None or len(df) < 50:
        return "NEUTRAL", 0, ["Insufficient data points for full technical indicator evaluation."]

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    score = 0
    signals = []

    # 1. RSI Signal
    rsi = latest.get("RSI_14", 50)
    if not np.isnan(rsi):
        if rsi < 30:
            score += 30
            signals.append(f"🟢 **RSI Oversold ({rsi:.1f})**: High reversal / buying pressure potential.")
        elif rsi > 70:
            score -= 30
            signals.append(f"🔴 **RSI Overbought ({rsi:.1f})**: Market potentially extended, risk of pullback.")
        elif 45 <= rsi <= 55:
            signals.append(f"⚪ **RSI Neutral ({rsi:.1f})**: Balanced momentum.")
        elif rsi > 55:
            score += 15
            signals.append(f"🟢 **RSI Bullish Momentum ({rsi:.1f})**: Positive trending momentum.")
        else:
            score -= 15
            signals.append(f"🔴 **RSI Bearish Momentum ({rsi:.1f})**: Negative trending momentum.")

    # 2. Moving Average Trend (Close vs EMA 20 & SMA 50)
    close_val = latest["Close"]
    ema_20 = latest.get("EMA_20", np.nan)
    sma_50 = latest.get("SMA_50", np.nan)

    if not np.isnan(ema_20):
        if close_val > ema_20:
            score += 20
            signals.append(f"🟢 **Price > EMA 20**: Short-term trend is upward.")
        else:
            score -= 20
            signals.append(f"🔴 **Price < EMA 20**: Short-term trend is downward.")

    if not np.isnan(sma_50):
        if close_val > sma_50:
            score += 20
            signals.append(f"🟢 **Price > SMA 50**: Medium-term bullish structural support.")
        else:
            score -= 20
            signals.append(f"🔴 **Price < SMA 50**: Medium-term resistance overhead.")

    # 3. MACD Crossover Signal
    macd = latest.get("MACD", 0)
    macd_signal = latest.get("MACD_Signal", 0)
    prev_macd = prev.get("MACD", 0)
    prev_macd_signal = prev.get("MACD_Signal", 0)

    if not np.isnan(macd) and not np.isnan(macd_signal):
        if prev_macd <= prev_macd_signal and macd > macd_signal:
            score += 30
            signals.append("🚀 **Fresh MACD Bullish Crossover**: MACD line surged above signal line!")
        elif prev_macd >= prev_macd_signal and macd < macd_signal:
            score -= 30
            signals.append("⚠️ **Fresh MACD Bearish Crossover**: MACD line crossed below signal line.")
        elif macd > macd_signal:
            score += 15
            signals.append("🟢 **MACD Above Signal**: Bullish continuation.")
        else:
            score -= 15
            signals.append("🔴 **MACD Below Signal**: Bearish continuation.")

    # 4. Bollinger Bands Positioning
    bb_upper = latest.get("BB_Upper", np.nan)
    bb_lower = latest.get("BB_Lower", np.nan)
    if not np.isnan(bb_upper) and not np.isnan(bb_lower):
        if close_val >= bb_upper:
            signals.append("⚠️ **Testing Upper Bollinger Band**: Near upper volatility boundary.")
        elif close_val <= bb_lower:
            score += 15
            signals.append("🟢 **Testing Lower Bollinger Band**: Value rebound zone.")

    # Determine overall rating
    if score >= 50:
        overall = "STRONG BUY"
    elif score >= 20:
        overall = "BUY"
    elif score <= -50:
        overall = "STRONG SELL"
    elif score <= -20:
        overall = "SELL"
    else:
        overall = "NEUTRAL"

    return overall, score, signals
