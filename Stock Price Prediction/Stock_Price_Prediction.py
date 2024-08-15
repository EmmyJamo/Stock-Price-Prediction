# Import necessary libraries
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from ta import add_all_ta_features

# Step 1: Data Collection
symbol = 'AAPL'
data = yf.download(symbol, start='2010-01-01', end='2023-01-01')

# Step 2: Feature Engineering
# Adding technical indicators using the `ta` library
data = add_all_ta_features(
    df=data, open="Open", high="High", low="Low", close="Adj Close", volume="Volume", fillna=True)

# Adding time-based features
data['DayOfWeek'] = data.index.dayofweek
data['Month'] = data.index.month

# Adding Lagged Returns
data['Lag_1'] = data['Adj Close'].shift(1)
data['Lag_5'] = data['Adj Close'].shift(5)

# Adding Volume-Based Features
data['Volume_SMA_20'] = data['Volume'].rolling(window=20).mean()
data['Volume_SMA_50'] = data['Volume'].rolling(window=50).mean()

# Adding Volatility Features
data['Volatility_20'] = data['Adj Close'].rolling(window=20).std()
data['Volatility_50'] = data['Adj Close'].rolling(window=50).std()

# Target variable: predict next day's adjusted close price
data['Returns'] = data['Adj Close'].pct_change()
data['SMA_20'] = data['Adj Close'].rolling(window=20).mean()
data['SMA_50'] = data['Adj Close'].rolling(window=50).mean()

data.dropna(inplace=True)

# Step 3: Data Normalization
features = [
    'Returns', 'SMA_20', 'SMA_50', 'momentum_rsi', 'volatility_bbm', 'trend_macd', 
    'DayOfWeek', 'Month', 'Lag_1', 'Lag_5', 'Volume_SMA_20', 'Volume_SMA_50', 
    'Volatility_20', 'Volatility_50'
]
data[features] = (data[features] - data[features].mean()) / data[features].std()

# Display the first few rows of the prepared dataset to inspect the data
print(data.head())

# Step 4: Prepare Data for Modeling
X = data[features]
y = data['Adj Close'].shift(-1)  # Predict next day's price

X = X[:-1]
y = y[:-1]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the Random Forest Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions and evaluate the model's performance
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

train_rmse = mean_squared_error(y_train, y_pred_train, squared=False)
test_rmse = mean_squared_error(y_test, y_pred_test, squared=False)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print(f"Train RMSE: {train_rmse:.2f}, R2: {train_r2:.2f}")
print(f"Test RMSE: {test_rmse:.2f}, R2: {test_r2:.2f}")

# Step 6: Hyperparameter Tuning using Grid Search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Train the model using the best hyperparameters
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

# Evaluate the tuned model's performance
best_rmse = mean_squared_error(y_test, y_pred_best, squared=False)
best_r2 = r2_score(y_test, y_pred_best)

print(f"Best Test RMSE: {best_rmse:.2f}, R2: {best_r2:.2f}")

# Step 7: Predict Future Outcomes
future_data = yf.download(symbol, start='2023-01-01', end='2024-01-01')

# Adding technical indicators and time-based features for future data
future_data = add_all_ta_features(
    df=future_data, open="Open", high="High", low="Low", close="Adj Close", volume="Volume", fillna=True)
future_data['DayOfWeek'] = future_data.index.dayofweek
future_data['Month'] = future_data.index.month
future_data['Lag_1'] = future_data['Adj Close'].shift(1)
future_data['Lag_5'] = future_data['Adj Close'].shift(5)
future_data['Volume_SMA_20'] = future_data['Volume'].rolling(window=20).mean()
future_data['Volume_SMA_50'] = future_data['Volume'].rolling(window=50).mean()
future_data['Volatility_20'] = future_data['Adj Close'].rolling(window=20).std()
future_data['Volatility_50'] = future_data['Adj Close'].rolling(window=50).std()

# Recalculate the same features used in the training data
future_data['Returns'] = future_data['Adj Close'].pct_change()
future_data['SMA_20'] = future_data['Adj Close'].rolling(window=20).mean()
future_data['SMA_50'] = future_data['Adj Close'].rolling(window=50).mean()

# Drop any rows with NaN values that resulted from rolling calculations
future_data.dropna(inplace=True)

# Standardize features for prediction
future_features = future_data[features]
future_features = (future_features - X.mean()) / X.std()

future_predictions = best_model.predict(future_features)

# Step 8: Visualize Predictions
plt.figure(figsize=(10, 6))
plt.plot(future_data.index, future_data['Adj Close'], label='Actual Price')
plt.plot(future_data.index, future_predictions, label='Predicted Price', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Future Price Predictions for AAPL')
plt.legend()
plt.show()
