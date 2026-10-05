# 🚀 CoinSight – Crypto & Bitcoin AI Analytics Terminal

A full-stack, institutional-grade cryptocurrency analytics and price prediction terminal powered by **Streamlit**, **Plotly**, and **Deep Learning (Stacked LSTM Neural Networks)**.

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.54-FF4B4B?logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Dark-3F4F75?logo=plotly&logoColor=white)

---

## 📊 Overview

**CoinSight** is an AI-driven cryptocurrency market intelligence and price forecasting system. It combines real-time technical indicators, quantitative risk modeling, dynamic confidence intervals, and deep learning (Stacked LSTM) to analyze market dynamics and forecast price trends across Bitcoin and major cryptocurrencies.

---

## 🌟 Key Features

1. **Live Real-Time Market Data**:
   - Live price feeds, 24h High/Low, 24h Trading Volume, and 7-day Annualized Volatility.
   - Multi-crypto selector: **Bitcoin (BTC)**, **Ethereum (ETH)**, **Solana (SOL)**, **Binance Coin (BNB)**, **Cardano (ADA)**, **Ripple (XRP)**, or any custom Yahoo Finance ticker.
   - Automatic caching with offline fallback resilience.

2. **Interactive Candlestick & Technical Analysis**:
   - Candlestick price action with volume bars.
   - Overlays: **Bollinger Bands (20, 2σ)**, **SMA (20, 50)**, and **EMA (20)**.
   - Separate momentum oscillators: **RSI (14)** with 70/30 overbought/oversold bands, and **MACD (12, 26, 9)** with color-coded histogram.
   - **Automated AI Composite Signal**: Real-time momentum scoring (`STRONG BUY`, `BUY`, `NEUTRAL`, `SELL`, `STRONG SELL`) based on technical indicator confluence.

3. **Deep Learning AI Price Forecasting**:
   - Stacked **LSTM (Long Short-Term Memory)** neural network forecasting 3 to 60 days into the future.
   - Dynamic **Confidence Interval Bands (95% / 90% / 80%)** representing market volatility uncertainty.
   - Day-by-day projected prices, expected dollar returns, and ROI percentage.
   - Downloadable forecast CSV report.

4. **Investment & ROI Simulator**:
   - Interactive portfolio simulation: input investment capital ($) and test outcomes.
   - Generates simulated equity curves with best-case (upper bound) and worst-case (lower bound) drawdown risk scenarios.

5. **Model Diagnostics & In-App Retraining**:
   - Displays evaluation metrics on recent test windows: **RMSE**, **MAE**, **MAPE**, and **Directional Accuracy (%)**.
   - Single-click in-app model retraining on latest historical market data with live progress status.

6. **Cyberpunk FinTech Design**:
   - Modern dark glassmorphic UI, custom typography (`Inter` & `JetBrains Mono`), animated live pulse dots, and responsive KPI metric cards.

---

## 📁 Project Architecture

```
bitcoin/
├── app.py                      # Main Streamlit dashboard application
├── train_model.py              # Standalone training script for LSTM model
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── assets/
│   └── style.css               # Modern dark fintech glassmorphism design
├── components/
│   ├── data_loader.py          # Real-time yfinance fetching, caching, KPIs
│   ├── indicators.py           # Technical indicators & composite signal engine
│   ├── charts.py               # Interactive Plotly candlestick & forecast charts
│   ├── forecast.py             # LSTM prediction engine & accuracy metrics
│   ├── simulator.py            # Portfolio ROI & equity curve simulator
│   └── sidebar.py              # User configuration & asset selection
├── data/
│   └── crypto_data.csv         # Cached historical data
├── models/
│   └── lstm_crypto_model.keras # Trained LSTM neural network model
└── outputs/
    └── training_loss.png       # Model training vs validation loss curve
```

---

## ⚙️ Installation

```bash
git clone https://github.com/DhanushTech882/bitcoin.git
cd bitcoin
pip install -r requirements.txt
```

---

## 🚀 How to Run the Project

### Recommended (Direct Execution via Virtual Environment)

From the project root directory in **PowerShell**:

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

### Alternative (With Virtual Environment Activation)

#### In PowerShell:
```powershell
.\venv\Scripts\Activate.ps1
streamlit run app.py
```

#### In Command Prompt (CMD):
```cmd
.\venv\Scripts\activate.bat
streamlit run app.py
```

Once started, open your browser and navigate to:
👉 **`http://localhost:8501`**

---

## 🧠 Neural Network Model Architecture

- **Input Shape**: `(Lookback Window = 60, 1 Feature)`
- **Layer 1**: `LSTM(64 units, return_sequences=True)`
- **Layer 2**: `Dropout(0.20)`
- **Layer 3**: `LSTM(64 units, return_sequences=False)`
- **Layer 4**: `Dropout(0.20)`
- **Layer 5**: `Dense(32 units, activation='relu')`
- **Output**: `Dense(1 unit, linear)`
- **Loss**: Mean Squared Error (MSE)
- **Optimizer**: Adam

---

## ☁️ Cloud Deployment

### 1. Streamlit Community Cloud (Recommended & Free)
Streamlit Cloud natively hosts the full interactive dashboard and LSTM neural network inference:
1. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
2. Click **New app**.
3. Select Repository: `DhanushTech882/bitcoin`, Branch: `main`, Main file path: `app.py`.
4. Click **Deploy!**

### 2. Vercel Web Portal
This repository includes a static web portal and market ticker configured for Vercel:
- Automatically deployed using `.vercelignore` and `vercel.json` without Serverless Function conflicts.

### 3. Docker Container
Build and run anywhere with Docker:
```bash
docker build -t coinsight .
docker run -p 8501:8501 coinsight
```

### 4. Render
Connect this repository to [render.com](https://render.com) as a Web Service. The included `render.yaml` sets up the build and start commands automatically.

---

## 👨‍💻 Author

**Dhanush Tech**
- GitHub: [@DhanushTech882](https://github.com/DhanushTech882)
- Repository: [https://github.com/DhanushTech882/bitcoin](https://github.com/DhanushTech882/bitcoin)

