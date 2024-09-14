import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.decomposition import PCA
import joblib

# Load your DataFrame
data = pd.read_csv(r'C:\Users\hp\Desktop\Thesis\1804010_Thunderstorm&Lightning Dataset.csv')
data.drop(['Date', 'District', 'Longitude', 'Latitude'], axis=1, inplace=True)
data.fillna(0, inplace=True)

# Separate features (X) and target (y)
X = data.drop(data.columns[-1], axis=1).values
y = data[data.columns[-1]].values

# Scaling the features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_X = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=0.95)  # Keep 95% of the variance
reduced_X = pca.fit_transform(scaled_X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(reduced_X, y, test_size=0.2, random_state=42)

# Define the model
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.2))  # Add dropout for regularization
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))  # Single output for regression

# Compile the model
optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Fit the model
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), batch_size=32, verbose=1)

# Save the trained model
model.save('your_model_with_pca.h5')

# Save the PCA transformer
joblib.dump(pca, 'pca_transformer.save')

# Save the scaler
joblib.dump(scaler, 'scaler.save')

# Evaluate the model
train_loss = model.evaluate(X_train, y_train, verbose=0)
val_loss = model.evaluate(X_test, y_test, verbose=0)
train_rmse = np.sqrt(train_loss)
val_rmse = np.sqrt(val_loss)

print(f'Training RMSE: {train_rmse:.4f}')
print(f'Validation RMSE: {val_rmse:.4f}')

# Example to use the model for prediction
# Select a sample from the test set to predict
new_data = X_test[0].reshape(1, -1)  # Ensure the shape is (1, n_features)

# Make a prediction
prediction = model.predict(new_data)

# Inverse transform the prediction if necessary
# If the target variable was scaled, you would inverse transform the predicted value here.
# However, in this example, the target variable was not scaled.
print(f'Prediction: {prediction[0][0]:.4f}')

# Visualization of PCA Components
explained_variance = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance)

plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.5, align='center', label='individual explained variance')
plt.step(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, where='mid', label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.title('Explained Variance by Principal Components')
plt.show()

# Determine the importance of each original feature in the first principal component
loadings = pca.components_[0]
# Get feature names
feature_names = data.columns[:-1]

# Create a DataFrame for visualization
feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': np.abs(loadings)})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'], align='center')
plt.xlabel('Importance')
plt.title('Feature Importance in the First Principal Component')
plt.gca().invert_yaxis()
plt.show()
