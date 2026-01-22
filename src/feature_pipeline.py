import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from src.config import PROJECT_DIR


def build_feature_pipeline():
    cat_cols = ["batter", "venue", "team1", "team2"]
    num_cols = [
        "runs_last_5_avg",
        "runs_last_10_avg",
        "career_runs_avg",
        "venue_runs_avg",
        "pvt_runs_avg",
        "pvp_runs_avg",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor)
    ])

    return pipeline


def save_pipeline():
    pipeline = build_feature_pipeline()

    out_path = PROJECT_DIR / "models" / "feature_pipeline.pkl"
    out_path.parent.mkdir(exist_ok=True)

    joblib.dump(pipeline, out_path)
    print("✅ Saved:", out_path)


if __name__ == "__main__":
    save_pipeline()
