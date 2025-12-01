from matplotlib import pyplot as plt
from skimage.metrics import mean_squared_error
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

# Load dataset
data_frame = pd.read_csv("data/healthcare_dataset.csv")

# Drop missing values
data_frame.dropna(inplace=True)

# Columns to encode
categorical_features = ['Gender', 'Blood Type', 'Medical Condition', 'Admission Type', 'Medication']

# Column transformer for one-hot encoding
ct = ColumnTransformer(
        transformers=[('encoder', OneHotEncoder(sparse_output=False), categorical_features)],
        remainder='passthrough'  # keep remaining columns (like Age) as-is
)

# Features
X = data_frame[categorical_features + ['Age']]

# Target - encode numerically
le = LabelEncoder()
y = le.fit_transform(data_frame['Test Results'].values)  # Converts categories to 0,1,...

# Apply column transformer
X_encoded = ct.fit_transform(X)

# Scale Age column (last column after one-hot encoding)
scaler = MinMaxScaler()
X_encoded[:, -1] = scaler.fit_transform(X_encoded[:, -1].reshape(-1, 1)).flatten()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=1)

# Linear regression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predictions
y_pred = regressor.predict(X_test)

# Compute MSE
mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)

# Convert regression output to nearest class
y_pred_class = np.round(y_pred).astype(int)

# Clip predictions to valid classes (0,1,2)
y_pred_class = np.clip(y_pred_class, 0, 2)

# Compute accuracy percentage
accuracy = np.mean(y_pred_class == y_test) * 100
print("Accuracy:", accuracy, "%")

# Scatter plot
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Test Results (0,1,2)")
plt.ylabel("Predicted Value (Linear Regression)")
plt.title("Actual vs Predicted")

# Line y = x for reference
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linewidth=2)

plt.show()