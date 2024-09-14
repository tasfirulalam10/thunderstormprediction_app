import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, GRU
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
# Load your DataFrame (this line is just for example, replace it with your actual DataFrame loading code)
data = pd.read_csv(r'C:\Users\hp\Desktop\Thesis\DistrictData\Dhaka .csv')
data
data['Date'] = pd.to_datetime(data['Date'])
data.drop(['District', 'Longitude', 'Latitude'], axis=1, inplace=True)
data.set_index('Date')[['Events','Total Precipitation','Cape','2m_dewpoint_temperature','2m_temperature','convective_inhibition','convective_precipitation','convective_rain_rate','evaporation','surface_pressure','total_totals_index','total_cloud_cover','k_index','10m_v_component_of_wind','10m_u_component_of_wind']].plot(subplots=True,figsize=(10, 10))

data.fillna(0, inplace=True)
data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data.drop(['Date'], axis=1))
# Create sequences for LSTM
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data)-seq_length):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 10  # A common choice for time series
X, y = create_sequences(scaled_data, seq_length)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

# Define the model
model = Sequential()
model.add(GRU(128, activation='selu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))  # Add dropout for regularization
model.add(GRU(64, activation='selu'))
model.add(Dropout(0.2))  # Add dropout for regularization
model.add(Dense(y_train.shape[1], activation='linear'))  # Ensure activation is linear for regression

# Define the optimizer with gradient clipping
optimizer = Adam(learning_rate=0.001, clipnorm=1.0)

# Compile the model
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Print model summary
model.summary()

# Early stopping to prevent overfitting
#early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Fit the model
history = model.fit(X_train, y_train, epochs=60, validation_data=(X_test, y_test), batch_size=32, verbose=1)#, callbacks=[early_stopping])

# Evaluate the model
train_loss = model.evaluate(X_train, y_train, verbose=0)
val_loss = model.evaluate(X_test, y_test, verbose=0)

# Calculate RMSE
train_rmse = np.sqrt(train_loss)
val_rmse = np.sqrt(val_loss)

print(f'Training RMSE: {train_rmse:.4f}')
print(f'Validation RMSE: {val_rmse:.4f}')

# Plotting the training and validation loss
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot actual vs. predicted values
y_pred = model.predict(X_test)

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

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    epsilon = 1e-10  # Small value to prevent division by zero
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon)))# * 100

mape= mean_absolute_percentage_error(y_test, y_pred)
    # Print evaluation metrics
print("Root Mean Squared Error (RMSE):", rmse*10)
print("R2 Score:", r2*10)
print("Mean Absolute Error (MAE):", mae*10)
print("Mean Absolute Percentage Error (MAPE):", mape*10)

