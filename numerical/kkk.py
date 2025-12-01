import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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
y = le.fit_transform(data_frame['Test Results'].values)  # Converts categories to 0,1,2

# Apply column transformer
X_encoded = ct.fit_transform(X)

# Scale Age column (last column after one-hot encoding)
scaler = MinMaxScaler()
X_encoded[:, -1] = scaler.fit_transform(X_encoded[:, -1].reshape(-1, 1)).flatten()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=1)

# K-Nearest Neighbors classifier
knn = KNeighborsClassifier(n_neighbors=5)  # You can tune n_neighbors
knn.fit(X_train, y_train)

# Predictions
y_pred = knn.predict(X_test)

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred) * 100
print("Accuracy:", accuracy, "%")

# Optional: confusion matrix and detailed report
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
