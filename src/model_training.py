import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "processed" / "dataset.csv"

df = pd.read_csv(DATA_PATH)

print("Loaded dataset:", df.shape)

# Target 1: next_match_runs
target = "next_match_runs"

# Features
X = df.drop(columns=["next_match_runs", "next_match_wickets"])
y = df[target]

# Train-test split (time-series aware)
# We'll split by date sorting to avoid leakage
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.sort_values("date")

X = df.drop(columns=["next_match_runs", "next_match_wickets"])
y = df[target]

split_index = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

print("Train:", X_train.shape, "Test:", X_test.shape)

# Columns
cat_cols = ["player", "season", "venue", "city"]
num_cols = [c for c in X.columns if c not in cat_cols and c != "date"]

# Preprocessor
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ],
    remainder="drop"
)

def evaluate_model(name, model, X_test, y_test):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)

    print(f"\n{name}")
    print("MAE :", mae)
    print("RMSE:", rmse)
    print("R2  :", r2)

# Model 1: Linear Regression
lr_model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", LinearRegression())
])

lr_model.fit(X_train, y_train)
evaluate_model("Linear Regression", lr_model, X_test, y_test)

# Model 2: Random Forest
rf_model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ))
])

rf_model.fit(X_train, y_train)
evaluate_model("Random Forest", rf_model, X_test, y_test)
