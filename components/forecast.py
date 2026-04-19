import numpy as np
from sklearn.preprocessing import MinMaxScaler

def generate_forecast(model, data, lookback, future_days):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    last_days = scaled_data[-lookback:].reshape(1, lookback, 1)
    predictions = []

    for _ in range(future_days):
        next_price = model.predict(last_days, verbose=0)[0][0]
        predictions.append(next_price)
        last_days = np.append(last_days[:, 1:, :], [[[next_price]]], axis=1)

    predictions = scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1)
    )

    return predictions
