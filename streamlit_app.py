import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Cricket Player Performance Prediction",
    layout="wide"
)

st.title("🏏 Cricket Player Performance Prediction (IPL)")
st.write("Predict **Runs (Batsman)** or **Wickets (Bowler)** using Machine Learning")

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
batsman_df = pd.read_csv("dataset.csv")
bowler_df = pd.read_csv("dataset_bowler.csv")

batsman_df["date"] = pd.to_datetime(batsman_df["date"])
bowler_df["date"] = pd.to_datetime(bowler_df["date"])

# -------------------------------------------------
# LOAD MODELS & PIPELINES
# -------------------------------------------------
batsman_model = joblib.load("xgb_model.joblib")
batsman_preprocessor = joblib.load("feature_pipeline.pkl")

bowler_model = joblib.load("bowler_model.joblib")
bowler_preprocessor = joblib.load("feature_pipeline_bowler.pkl")

# -------------------------------------------------
# SIDEBAR INPUTS
# -------------------------------------------------
st.sidebar.header("Match Inputs")

player_type = st.sidebar.radio(
    "Select Player Type",
    ["Batsman", "Bowler"]
)

# -------------------------------------------------
# BATSMAN INPUTS
# -------------------------------------------------
if player_type == "Batsman":
    player = st.sidebar.selectbox(
        "Select Batsman",
        sorted(batsman_df["batsman"].unique())
    )

    venue = st.sidebar.selectbox(
        "Venue",
        sorted(batsman_df["venue"].unique())
    )

    team = st.sidebar.selectbox(
        "Batting Team",
        sorted(batsman_df["batting_team"].unique())
    )

# -------------------------------------------------
# BOWLER INPUTS
# -------------------------------------------------
else:
    player = st.sidebar.selectbox(
        "Select Bowler",
        sorted(bowler_df["bowler"].unique())
    )

    venue = st.sidebar.selectbox(
        "Venue",
        sorted(bowler_df["venue"].unique())
    )

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------
if st.sidebar.button("Predict Performance"):

    # ==========================
    # BATSMAN PREDICTION
    # ==========================
    if player_type == "Batsman":
        row = (
            batsman_df[batsman_df["batsman"] == player]
            .sort_values("date")
            .iloc[-1]
        )

        input_df = pd.DataFrame([{
            "avg_runs_last_5": row["avg_runs_last_5"],
            "avg_runs_last_10": row["avg_runs_last_10"],
            "venue_avg_runs": row["venue_avg_runs"],
            "career_avg_runs": row["career_avg_runs"],
            "career_matches": row["career_matches"],
            "pvt_avg_runs": row["pvt_avg_runs"],
            "pvp_avg_runs": row["pvp_avg_runs"],
            "venue": venue,
            "batting_team": team
        }])

        X = batsman_preprocessor.transform(input_df)
        prediction = batsman_model.predict(X)[0]

        st.subheader("📊 Predicted Runs")
        st.metric("Expected Runs", f"{prediction:.1f}")

        # -------- Player Form --------
        st.subheader("📈 Player Form (Last 10 Matches)")
        recent = (
            batsman_df[batsman_df["batsman"] == player]
            .sort_values("date")
            .tail(10)
        )

        fig, ax = plt.subplots()
        ax.plot(recent["date"], recent["runs"], marker="o")
        ax.set_xlabel("Date")
        ax.set_ylabel("Runs")
        st.pyplot(fig)

        # -------- SHAP --------
        st.subheader("🔍 Feature Importance (SHAP)")
        explainer = shap.TreeExplainer(batsman_model)
        shap_values = explainer.shap_values(X)

        fig2 = plt.figure()
        shap.summary_plot(
            shap_values,
            X,
            feature_names=batsman_preprocessor.get_feature_names_out(),
            plot_type="bar",
            show=False
        )
        st.pyplot(fig2)

    # ==========================
    # BOWLER PREDICTION
    # ==========================
    else:
        row = (
            bowler_df[bowler_df["bowler"] == player]
            .sort_values("date")
            .iloc[-1]
        )

        input_df = pd.DataFrame([{
            "balls": row["balls"],
            "runs_conceded": row["runs_conceded"],
            "economy": row["economy"],
            "venue": venue
        }])

        X = bowler_preprocessor.transform(input_df)
        prediction = bowler_model.predict(X)[0]

        st.subheader("📊 Predicted Wickets")
        st.metric("Expected Wickets", f"{prediction:.2f}")

        # -------- Bowler Form --------
        st.subheader("📈 Bowler Form (Last 10 Matches)")
        recent = (
            bowler_df[bowler_df["bowler"] == player]
            .sort_values("date")
            .tail(10)
        )

        fig, ax = plt.subplots()
        ax.plot(recent["date"], recent["wickets"], marker="o")
        ax.set_xlabel("Date")
        ax.set_ylabel("Wickets")
        st.pyplot(fig)
