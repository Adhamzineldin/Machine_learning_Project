import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except:
    from sklearn.ensemble import RandomForestClassifier
    CATBOOST_AVAILABLE = False


# ============================
# 1. LOAD DATA
# ============================
df = pd.read_csv("data/healthcare_dataset.csv")

# ============================
# 2. CLEAN TEXT
# ============================
for col in ["Name", "Doctor", "Hospital"]:
    df[col] = df[col].astype(str).str.lower().str.strip()

df["Gender"] = df["Gender"].astype(str).str.lower().str.strip()

# ============================
# 3. DATE ENGINEERING
# ============================
df["Date of Admission"] = pd.to_datetime(df["Date of Admission"], errors="coerce")
df["Discharge Date"] = pd.to_datetime(df["Discharge Date"], errors="coerce")

df["length_of_stay"] = (df["Discharge Date"] - df["Date of Admission"]).dt.days
df["admission_month"] = df["Date of Admission"].dt.month
df["admission_weekday"] = df["Date of Admission"].dt.weekday

df = df.sort_values("Date of Admission")

# ============================
# 4. DROP LEAKAGE
# ============================
df = df.drop(columns=[
        "Name",
        "Doctor",
        "Hospital",
        "Room Number",
        "Date of Admission",
        "Discharge Date"
])

# ============================
# 5. TARGET ENCODING
# ============================
target_map = {
        "Normal": 0,
        "Abnormal": 1,
        "Inconclusive": 2
}
df["Test Results"] = df["Test Results"].map(target_map)

df = df.dropna(subset=["Test Results"])
df["Test Results"] = df["Test Results"].astype(int)

# ============================
# 6. FEATURES
# ============================
y = df["Test Results"]
X = df.drop(columns=["Test Results"])

cat_features = X.select_dtypes(include=["object"]).columns.tolist()
num_features = X.select_dtypes(exclude=["object"]).columns.tolist()

# ============================
# 7. PREPROCESSOR (FOR SKLEARN)
# ============================
preprocessor = ColumnTransformer(
        transformers=[
                ("num", RobustScaler(), num_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
        ]
)

# ============================
# 8. TIME-BASED SPLIT
# ============================
tscv = TimeSeriesSplit(n_splits=5)
train_index, test_index = list(tscv.split(X))[-1]

X_train, X_test = X.iloc[train_index], X.iloc[test_index]
y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# ============================
# 9. MODEL
# ============================
if CATBOOST_AVAILABLE:
    print("Using CatBoost")

    class_counts = y_train.value_counts().sort_index()
    class_weights = (class_counts.max() / class_counts).values.tolist()

    model = CatBoostClassifier(
            iterations=600,
            depth=6,
            learning_rate=0.05,
            loss_function="MultiClass",
            eval_metric="TotalF1",
            class_weights=class_weights,
            verbose=0
    )

    model.fit(
            X_train,
            y_train,
            cat_features=[X.columns.get_loc(c) for c in cat_features]
    )

    y_pred = model.predict(X_test).flatten()

else:
    print("Using RandomForest + OneHotEncoding")

    rf = RandomForestClassifier(
            n_estimators=500,
            max_depth=12,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
    )

    pipeline = Pipeline([
            ("prep", preprocessor),
            ("model", rf)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

# ============================
# 10. EVALUATION
# ============================
accuracy = accuracy_score(y_test, y_pred)

report = classification_report(
        y_test,
        y_pred,
        target_names=["Normal", "Abnormal", "Inconclusive"]
)

cm = confusion_matrix(y_test, y_pred)

# ============================
# 11. OUTPUT
# ============================
print("\n==============================")
print(" MEDICAL TEST RESULT PREDICTOR")
print("==============================")
print(f"\nAccuracy: {accuracy:.4f}\n")

print("Classification Report:")
print(report)

print("Confusion Matrix:")
print(cm)
