import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import os

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

CRYPTO_NAME = "BTC-USD"
START_DATE = "2018-01-01"
LOOKBACK = 60
EPOCHS = 10
BATCH_SIZE = 32

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

print("Downloading Bitcoin data...")
df = yf.download(CRYPTO_NAME, start=START_DATE, progress=False)

df = df[["Close"]].dropna()
df.to_csv("data/crypto_data.csv")

data = df["Close"].values.reshape(-1, 1)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

X, y = [], []
for i in range(LOOKBACK, len(scaled_data)):
    X.append(scaled_data[i-LOOKBACK:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)
X = X.reshape(X.shape[0], X.shape[1], 1)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(LOOKBACK, 1)),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(32),
    Dense(1)
])

model.compile(optimizer="adam", loss="mean_squared_error")

print("Training model...")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test)
)

model.save("models/lstm_crypto_model.keras")

plt.figure(figsize=(10,5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.grid()
plt.savefig("outputs/training_loss.png")

print("Model training completed!")
