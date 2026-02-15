import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# =============================
# Paths
# =============================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"


# =============================
# Load Data
# =============================

@st.cache_data
def load_runs_data():
    return pd.read_csv(DATA_DIR / "dataset.csv")


@st.cache_data
def load_bowler_data():
    return pd.read_csv(DATA_DIR / "bowler_dataset.csv")


# =============================
# Load Models
# =============================

@st.cache_resource
def load_models():

    runs_model = joblib.load(
        MODELS_DIR / "xgb_runs_model.joblib"
    )

    wickets_model = joblib.load(
        MODELS_DIR / "xgb_wickets_model.joblib"
    )

    return runs_model, wickets_model


# =============================
# Init
# =============================

runs_df = load_runs_data()
bowler_df = load_bowler_data()

runs_model, wickets_model = load_models()


# =============================
# UI
# =============================

st.set_page_config(
    page_title="AI Cricket Predictor",
    page_icon="🏏"
)

st.title("🏏 AI Cricket Performance Predictor")

option = st.sidebar.radio(
    "Prediction Type",
    ["Runs (Batsman)", "Wickets (Bowler)"]
)


# =============================
# RUNS
# =============================

if option == "Runs (Batsman)":

    st.header("🏏 Runs Prediction")

    batter = st.selectbox(
        "Select Batter",
        sorted(runs_df["batter"].dropna().unique())
    )

    batter_data = (
        runs_df[runs_df["batter"] == batter]
        .sort_values("date")
        .iloc[-1:]
    )

    # Keep batter (pipeline needs it)
    X = batter_data.drop(
        columns=["match_id", "date", "target_next_runs"],
        errors="ignore"
    )

    pred = runs_model.predict(X)[0]

    st.success(
        f"Predicted Runs: {round(pred,2)}"
    )

    # Show available numeric stats
    st.subheader("Recent Performance")

    numeric_cols = batter_data.select_dtypes(
        include=["int64", "float64"]
    ).columns

    st.dataframe(batter_data[numeric_cols])


# =============================
# WICKETS
# =============================

else:

    st.header("🎯 Wickets Prediction")

    bowler = st.selectbox(
        "Select Bowler",
        sorted(bowler_df["bowler"].dropna().unique())
    )

    bowler_data = (
        bowler_df[bowler_df["bowler"] == bowler]
        .sort_values("date")
        .iloc[-1:]
    )

    # Remove categorical bowler (XGB can't handle)
    X = bowler_data.drop(
        columns=[
            "match_id",
            "bowler",
            "date",
            "target_next_wickets"
        ],
        errors="ignore"
    )

    pred = wickets_model.predict(X)[0]

    st.success(
        f"Predicted Wickets: {round(pred,2)}"
    )

    st.subheader("Recent Bowling Stats")

    numeric_cols = bowler_data.select_dtypes(
        include=["int64", "float64"]
    ).columns

    st.dataframe(bowler_data[numeric_cols])


# =============================
# Footer
# =============================

st.markdown("---")
st.markdown("Developed by Abhishek | AI Cricket Project")