import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# Step 1: Data Collection
symbol = 'AAPL'
data = yf.download(symbol, start='2010-01-01', end='2023-01-01')

# Step 2: Feature Engineering
data['Returns'] = data['Adj Close'].pct_change()
data['SMA_20'] = data['Adj Close'].rolling(window=20).mean()
data['SMA_50'] = data['Adj Close'].rolling(window=50).mean()
data['Lag_1'] = data['Adj Close'].shift(1)
data.dropna(inplace=True)

# Step 3: Data Preparation for LSTM
# Use only 'Adj Close' for simplicity, but you can add more features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['Adj Close']])

# Create sequences of 60 time steps
def create_sequences(data, time_steps=60):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps])
        y.append(data[i+time_steps])
    return np.array(X), np.array(y)

time_steps = 60
X, y = create_sequences(scaled_data, time_steps)

# Split into train and test sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Reshape data to [samples, time_steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Step 4: Build LSTM Model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=25))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=20)

# Step 5: Predict Future Prices
# Predicting on the test data
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)  # Inverse scaling to get actual values

# Prepare test data for visualization
actual_prices = data['Adj Close'].values[-len(predictions):]
test_data = data.iloc[-len(predictions):]

# Step 6: Visualize Predictions
plt.figure(figsize=(10, 6))
plt.plot(test_data.index, actual_prices, label='Actual Price')
plt.plot(test_data.index, predictions, label='Predicted Price', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Future Price Predictions for AAPL using LSTM')
plt.legend()
plt.show()
