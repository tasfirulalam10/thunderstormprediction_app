import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from scikeras.wrappers import KerasRegressor
import matplotlib.pyplot as plt

# Load your DataFrame
data = pd.read_csv(r'C:\Users\hp\Desktop\Thesis\DistrictData\Dhaka .csv')

# Preprocess data
data['Date'] = pd.to_datetime(data['Date'])
data.drop(['District', 'Longitude', 'Latitude'], axis=1, inplace=True)
data.fillna(0, inplace=True)

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data.drop(['Date'], axis=1))

# Create sequences for GRU
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 10  # A common choice for time series
X, y = create_sequences(scaled_data, seq_length)

# Split data for initial search
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
def build_model(units=50, dropout_rate=0.2, learning_rate=0.001):
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
    model.add(GRU(units=units, activation='selu', return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(GRU(units=units, activation='selu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=y_train.shape[1], activation='linear'))

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
    return model

# Wrap the model using KerasRegressor
model = KerasRegressor(model=build_model, verbose=0)

# Define the hyperparameters grid
param_grid = {
    'model__units': [32, 64, 128],
    'model__dropout_rate': [0.2, 0.3, 0.4],
    'model__learning_rate': [0.001, 0.01, 0.1],
    'batch_size': [16, 32, 64],
    'epochs': [50, 100]
}

# Create GridSearchCV
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose=1)

# Perform Grid Search
grid_result = grid.fit(X_train, y_train)

# Summarize the results
print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")

# Evaluate the best model
best_model = grid_result.best_estimator_.model_

# Evaluate the model
train_loss = best_model.evaluate(X_train, y_train, verbose=0)
val_loss = best_model.evaluate(X_test, y_test, verbose=0)

# Calculate RMSE
train_rmse = np.sqrt(train_loss)
val_rmse = np.sqrt(val_loss)

print(f'Training RMSE: {train_rmse:.4f}')
print(f'Validation RMSE: {val_rmse:.4f}')

# Plot actual vs. predicted values
y_pred = best_model.predict(X_test)

# Inverse transform the scaled data to original scale
y_test_inverse = scaler.inverse_transform(y_test)
y_pred_inverse = scaler.inverse_transform(y_pred)

feature_index = 0

plt.figure(figsize=(10, 6))
plt.plot(y_test_inverse[:, feature_index], label='Actual')
plt.plot(y_pred_inverse[:, feature_index], label='Predicted')
plt.title('Actual vs. Predicted Values')
plt.xlabel('Time')
plt.ylabel('Events')
plt.legend()
plt.show()

# Calculate evaluation metrics
rmse = np.sqrt(mean_squared_error(y_test_inverse[:, feature_index], y_pred_inverse[:, feature_index]))
r2 = r2_score(y_test_inverse[:, feature_index], y_pred_inverse[:, feature_index])
mae = mean_absolute_error(y_test_inverse[:, feature_index], y_pred_inverse[:, feature_index])

def mean_absolute_percentage_error(y_true, y_pred):
    epsilon = 1e-10  # Small value to prevent division by zero
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon)))

mape = mean_absolute_percentage_error(y_test_inverse[:, feature_index], y_pred_inverse[:, feature_index])

# Print evaluation metrics
print("Root Mean Squared Error (RMSE):", rmse)
print("R2 Score:", r2)
print("Mean Absolute Error (MAE):", mae)
print("Mean Absolute Percentage Error (MAPE):", mape)
