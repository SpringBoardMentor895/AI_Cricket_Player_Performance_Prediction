import pandas as pd
import joblib
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


# ================= PATHS =================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

BATTER_PATH = DATA_DIR / "dataset.csv"
BOWLER_PATH = DATA_DIR / "bowler_dataset.csv"


# ================= LOAD DATA =================

batter_df = pd.read_csv(BATTER_PATH)
bowler_df = pd.read_csv(BOWLER_PATH)


# ================= BATTER MODEL =================

bat_features_cat = ["batter", "venue", "team1", "team2"]
bat_features_num = [
    "runs_last_5_avg",
    "runs_last_10_avg",
    "career_runs_avg",
    "career_sr",
    "venue_runs_avg",
    "pvt_runs_avg",
    "pvp_runs_avg"
]

TARGET_RUNS = "target_next_runs"


X_bat = batter_df[bat_features_cat + bat_features_num]
y_bat = batter_df[TARGET_RUNS]


preprocessor_bat = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), bat_features_cat),
        ("num", StandardScaler(), bat_features_num)
    ]
)


bat_model = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,

    tree_method="hist",
    predictor="cpu_predictor",
    gpu_id=None
)


bat_pipeline = Pipeline([
    ("prep", preprocessor_bat),
    ("model", bat_model)
])


print("Training Runs Model...")
bat_pipeline.fit(X_bat, y_bat)

joblib.dump(
    bat_pipeline,
    MODEL_DIR / "xgb_runs_model.joblib"
)

print("Runs model saved.")


# ================= BOWLER MODEL =================

wkt_features = [
    "overs",
    "runs",
    "economy",
    "wickets_last_5",
    "career_wickets_avg",
    "venue_wickets_avg"
]

TARGET_WKT = "target_next_wickets"


X_wkt = bowler_df[wkt_features]
y_wkt = bowler_df[TARGET_WKT]


wkt_model = XGBRegressor(
    n_estimators=250,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,

    tree_method="hist",
    predictor="cpu_predictor",
    gpu_id=None
)


print("Training Wickets Model...")
wkt_model.fit(X_wkt, y_wkt)

joblib.dump(
    wkt_model,
    MODEL_DIR / "xgb_wickets_model.joblib"
)

print("Wickets model saved.")

print("ALL MODELS RETRAINED SUCCESSFULLY ✅")