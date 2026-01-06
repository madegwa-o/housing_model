import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -----------------------------
# Configuration
# -----------------------------
DATA_PATH = "california_housing_train.csv"
TARGET_COLUMN = "median_house_value"
RANDOM_STATE = 42


# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)


# -----------------------------
# Define candidate models
# -----------------------------
models = {
    "LinearRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),
    "RandomForest": Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(
            n_estimators=150,
            max_depth=None,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])
}


# -----------------------------
# Train & evaluate
# -----------------------------
results = {}

for name, pipeline in models.items():
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    results[name] = {
        "mae": mean_absolute_error(y_test, preds),
        "mse": mean_squared_error(y_test, preds),
        "r2": r2_score(y_test, preds),
        "pipeline": pipeline
    }

    print(f"\n{name}")
    print(f"MAE: {results[name]['mae']:.2f}")
    print(f"MSE: {results[name]['mse']:.2f}")
    print(f"R2 : {results[name]['r2']:.4f}")


# -----------------------------
# Select best model
# -----------------------------
best_model_name = max(results, key=lambda k: results[k]["r2"])
best_pipeline = results[best_model_name]["pipeline"]

print("\nBest model selected:", best_model_name)


# -----------------------------
# Save artifacts
# -----------------------------
joblib.dump(best_pipeline, "best_model.pkl")
joblib.dump(X.columns.tolist(), "feature_names.pkl")

print("Artifacts saved:")
print("- best_model.pkl")
print("- feature_names.pkl")
