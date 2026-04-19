🚀 CoinSight – Hybrid Bitcoin Forecasting System
📊 Overview

CoinSight is a hybrid Bitcoin price prediction system that combines multiple statistical and deep learning models to improve forecasting accuracy.
The project integrates time-series models (ARIMA, Prophet) with deep learning (LSTM) and sentiment analysis to capture both market trends and public opinion.

The goal is to outperform individual models by leveraging their combined strengths.

🎯 Key Features

🔮 Hybrid prediction using LSTM + ARIMA + Prophet
🧠 Sentiment analysis on crypto-related text data
📈 Time-series forecasting with improved accuracy
📊 Visualization of Bitcoin price trends
⚡ Performance comparison between individual and hybrid models
🧠 Models Used
LSTM (Long Short-Term Memory) – captures sequential patterns
ARIMA – statistical time-series forecasting
Prophet (by Meta) – trend & seasonality modeling
Sentiment Analysis (NLP) – market mood detection

🛠️ Tech Stack
Programming Language: Python
Libraries:
Pandas, NumPy
Matplotlib, Seaborn
Scikit-learn
TensorFlow / Keras
Statsmodels
Prophet

📂 Project Structure

bitcoin/
│── data/                  # Historical Bitcoin datasets
│── models/                # Saved trained models
│── notebooks/             # Jupyter notebooks (experiments)
│── src/                   # Core source code
│   ├── preprocessing.py
│   ├── lstm_model.py
│   ├── arima_model.py
│   ├── prophet_model.py
│   ├── sentiment.py
│   └── hybrid_model.py
│── results/               # Output graphs & metrics
│── requirements.txt
│── main.py
│── README.md

⚙️ Installation
git clone https://github.com/DhanushTech882/bitcoin.git
cd bitcoin
pip install -r requirements.txt

▶️ Usage

Run the complete pipeline:

python main.py

Or explore notebooks:

jupyter notebook

📈 Workflow
Data collection (historical Bitcoin prices)
Data preprocessing & normalization
Train individual models (LSTM, ARIMA, Prophet)
Perform sentiment analysis on text data
Combine outputs into a hybrid model
Evaluate using RMSE, MAE, directional accuracy

📊 Results
Hybrid model shows better prediction accuracy than individual models
Improved trend prediction and volatility handling
More robust against market fluctuations

🔮 Future Enhancements
🌐 Real-time data integration (API-based)
📱 Web dashboard for live predictions
🤖 Advanced NLP (BERT-based sentiment analysis)
☁️ Deployment using cloud platforms

👨‍💻 Author

Dhanush Tech
GitHub: https://github.com/DhanushTech882
