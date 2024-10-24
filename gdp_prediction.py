import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 1. Mocking up the data
# ------------------------
# Generating mock data for GDP and various economic indicators
# np.random.seed ensures reproducibility of random numbers
np.random.seed(42)

# num_rows defines the number of data points (e.g., months or quarters)
num_rows = 200

# Creating a DataFrame with random values simulating economic data
data = pd.DataFrame({
    'GDP': np.random.normal(100, 5, num_rows),  # Mock GDP data, mean = 100, stddev = 5
    'Inflation': np.random.normal(2, 0.5, num_rows),  # Inflation around 2% (typical target)
    'Interest_Rates': np.random.normal(1.5, 0.3, num_rows),  # Interest rates around 1.5%
    'Unemployment_Rate': np.random.normal(5, 1, num_rows),  # Unemployment rate around 5%
    'Consumer_Sentiment': np.random.normal(80, 10, num_rows),  # Consumer sentiment index
    'Manufacturing_Index': np.random.normal(50, 5, num_rows),  # Manufacturing index (50 is neutral)
    'Stock_Market_Index': np.random.normal(3000, 100, num_rows)  # Stock market index around 3000 points
})

# Display the first few rows to understand the structure of the data
print(data.head())

# Splitting data into input features (X) and target (y)
X = data.drop('GDP', axis=1)  # Drop the GDP column from features
y = data['GDP']  # Target variable (GDP)

# Splitting the data into training and testing sets
# test_size=0.2 means 20% of data is reserved for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Random Forest Regression
# ---------------------------
# RandomForestRegressor is an ensemble method using multiple decision trees
# n_estimators: Number of trees in the forest (more trees = more accuracy, but also more computation)
# random_state: Fixes randomness to ensure reproducible results
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)  # Fit the model to the training data

# Predict GDP using the trained model on the test data
rf_preds = rf_model.predict(X_test)

# Compute RMSE (Root Mean Squared Error) to evaluate model accuracy
# RMSE gives us a measure of how far our predictions are from the actual values
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
print(f'Random Forest RMSE: {rf_rmse}')

# Plotting the actual vs predicted values for Random Forest
plt.figure(figsize=(10, 6))
plt.scatter(y_test, rf_preds, color='blue', label='Random Forest Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2, label='Perfect Prediction Line')
plt.xlabel('Actual GDP')
plt.ylabel('Predicted GDP')
plt.title('Random Forest: Actual vs Predicted GDP')
plt.legend()
plt.show()


# 3. XGBoost Regression
# ---------------------
# XGBRegressor is a gradient boosting method which optimizes errors by learning sequentially
# n_estimators: Number of boosting rounds (like how many trees we use in boosting)
# random_state: Ensures reproducibility of results
xgb_model = XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)  # Fit the model to the training data

# Predict GDP using the trained XGBoost model
xgb_preds = xgb_model.predict(X_test)

# Compute RMSE to evaluate accuracy
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
print(f'XGBoost RMSE: {xgb_rmse}')

# Plotting the actual vs predicted values for XGBoost
plt.figure(figsize=(10, 6))
plt.scatter(y_test, xgb_preds, color='green', label='XGBoost Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2, label='Perfect Prediction Line')
plt.xlabel('Actual GDP')
plt.ylabel('Predicted GDP')
plt.title('XGBoost: Actual vs Predicted GDP')
plt.legend()
plt.show()


# 4. LSTM (Long Short-Term Memory)
# ---------------------------------
# LSTM requires the input to be scaled and reshaped into 3D: [samples, timesteps, features]
# MinMaxScaler scales the data to a range between 0 and 1 (useful for neural networks)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)  # Scaling the data

# Splitting the scaled data into training and testing sets
X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Reshape the input data for LSTM (samples, timesteps, features)
X_train_lstm = X_train_lstm.reshape((X_train_lstm.shape[0], 1, X_train_lstm.shape[1]))
X_test_lstm = X_test_lstm.reshape((X_test_lstm.shape[0], 1, X_test_lstm.shape[1]))

# Build the LSTM model
# Sequential: Initializes the neural network model layer by layer
# LSTM(50): Adds an LSTM layer with 50 units (number of memory cells in the layer)
# input_shape=(timesteps, features): Input shape is set to 1 timestep and number of features
lstm_model = Sequential()
lstm_model.add(LSTM(50, activation='relu', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
# Dense(1): Adds a fully connected layer with 1 output (for predicting GDP)
lstm_model.add(Dense(1))
# Compile the model using mean squared error (mse) as the loss function and adam optimizer
lstm_model.compile(optimizer='adam', loss='mse') # The ADAM optimizer is an improvement over the Stochastic Gradient Boosting algorithm (to find optimal parameters like w and b in the function that minimize the loss function - in this case the MSE function (which is the difference between actual and predicted values essentially)

# Train the LSTM model on the training data
# epochs=50: The number of complete passes through the training data
lstm_model.fit(X_train_lstm, y_train_lstm, epochs=50, verbose=0)

# Predict GDP using the trained LSTM model
lstm_preds = lstm_model.predict(X_test_lstm)

# Compute RMSE to evaluate the LSTM model's accuracy
lstm_rmse = np.sqrt(mean_squared_error(y_test_lstm, lstm_preds))
print(f'LSTM RMSE: {lstm_rmse}')

# Plotting the actual vs predicted values for LSTM
plt.figure(figsize=(10, 6))
plt.scatter(y_test_lstm, lstm_preds, color='purple', label='LSTM Predictions')
plt.plot([min(y_test_lstm), max(y_test_lstm)], [min(y_test_lstm), max(y_test_lstm)], color='red', lw=2, label='Perfect Prediction Line')
plt.xlabel('Actual GDP')
plt.ylabel('Predicted GDP')
plt.title('LSTM: Actual vs Predicted GDP')
plt.legend()
plt.show()


# 5. Neural Network (Fully Connected)
# -----------------------------------
# A basic feedforward neural network (fully connected layers)
# Dense(64): Adds a hidden layer with 64 neurons and 'relu' activation function
# input_dim=X_train.shape[1]: Input dimension is the number of features
nn_model = Sequential()
nn_model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
# Add another hidden layer with 32 neurons
nn_model.add(Dense(32, activation='relu'))
# Add the output layer with 1 neuron for GDP prediction
nn_model.add(Dense(1))
# Compile the model using mean squared error (mse) as the loss function and adam optimizer
nn_model.compile(optimizer='adam', loss='mse')

# Train the neural network model
# epochs=50: The number of times the model will see the entire dataset during training
nn_model.fit(X_train, y_train, epochs=50, verbose=0)

# Predict GDP using the trained neural network model
nn_preds = nn_model.predict(X_test)

# Compute RMSE to evaluate the neural network's accuracy
nn_rmse = np.sqrt(mean_squared_error(y_test, nn_preds))
print(f'Neural Network RMSE: {nn_rmse}')

# Plotting the actual vs predicted values for the neural network
plt.figure(figsize=(10, 6))
plt.scatter(y_test, nn_preds, color='orange', label='Neural Network Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2, label='Perfect Prediction Line')
plt.xlabel('Actual GDP')
plt.ylabel('Predicted GDP')
plt.title('Neural Network: Actual vs Predicted GDP')
plt.legend()
plt.show()

