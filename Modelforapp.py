import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

import joblib

# Load your DataFrame
data = pd.read_csv(r'C:\Users\hp\Desktop\Thesis\DistrictData\Dhaka .csv')
data
data.drop(['Date', 'District', 'Longitude', 'Latitude'], axis=1, inplace=True)
data.fillna(0, inplace=True)

# Separate features (X) and target (y)
X = data.drop(data.columns[-1], axis=1)
y = data[data.columns[-1]]

# Assuming X has shape (None, 1, 14)
X = np.squeeze(X, axis=1)  # Fix: Remove squeeze if unnecessary

# Scaling the features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)
X_train_reshaped = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])  # Reshape for GRU

# Fix: Get the number of features directly from X_train
n_features = X_train.shape[2]

# Define the model
model = Sequential()
model.add(GRU(128, activation='gelu', return_sequences=True, input_shape=(1, n_features)))
model.add(Dropout(0.2))  # Add dropout for regularization
model.add(GRU(64, activation='gelu'))
model.add(Dropout(0.2))  # Add dropout for regularization
model.add(Dense(1, activation='linear'))  # Ensure activation is linear for regression (output dimension is 1 for single target variable)

# Compile the model
optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Fit the model
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), batch_size=32, verbose=1, callbacks=[early_stopping])

# Save the trained model
model.save('your_model.h5')

# Save the scaler
joblib.dump(scaler, 'scaler.save')

# Evaluate the model
train_loss = model.evaluate(X_train, y_train, verbose=0)
val_loss = model.evaluate(X_test, y_test, verbose=0)
train_rmse = np.sqrt(train_loss)
val_rmse = np.sqrt(val_loss)

print(f'Training RMSE: {train_rmse:.4f}')
print(f'Validation RMSE: {val_rmse:.4f}')
