# ============================================
# AI CRICKET PLAYER PERFORMANCE DASHBOARD
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from pathlib import Path


# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="AI Cricket Performance Predictor",
    layout="wide",
    page_icon="🏏"
)

st.title("🏏 AI-Based Cricket Player Performance Prediction")


# ============================================
# PATH SETUP
# ============================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"


BATTER_PATH = DATA_DIR / "dataset.csv"
BOWLER_PATH = DATA_DIR / "bowler_dataset.csv"

RUNS_MODEL_PATH = MODEL_DIR / "xgb_runs_model.joblib"
WKT_MODEL_PATH  = MODEL_DIR / "xgb_wickets_model.joblib"


# ============================================
# LOAD RESOURCES
# ============================================

@st.cache_data
def load_data():

    batter = pd.read_csv(BATTER_PATH)
    bowler = pd.read_csv(BOWLER_PATH)

    batter["date"] = pd.to_datetime(batter["date"])
    bowler["date"] = pd.to_datetime(bowler["date"])

    return batter, bowler


@st.cache_resource
def load_models():

    runs_model = joblib.load(RUNS_MODEL_PATH)
    wickets_model = joblib.load(WKT_MODEL_PATH)

    return runs_model, wickets_model


batter_df, bowler_df = load_data()
runs_model, wickets_model = load_models()


# ============================================
# SIDEBAR INPUTS
# ============================================

st.sidebar.header("🎯 Match Inputs")

batter = st.sidebar.selectbox(
    "Select Batter",
    sorted(batter_df["batter"].unique())
)

bowler = st.sidebar.selectbox(
    "Select Bowler",
    sorted(bowler_df["bowler"].unique())
)

venue = st.sidebar.selectbox(
    "Select Venue",
    sorted(batter_df["venue"].unique())
)

team1 = st.sidebar.selectbox(
    "Team 1",
    sorted(batter_df["team1"].unique())
)

team2 = st.sidebar.selectbox(
    "Team 2",
    sorted(batter_df["team2"].unique())
)

predict_btn = st.sidebar.button("🚀 Predict Performance")


# ============================================
# MAIN DASHBOARD
# ============================================

if predict_btn:

    # ======================================
    # BATTER SECTION
    # ======================================

    bat_hist = batter_df[
        batter_df["batter"] == batter
    ].sort_values("date")

    last_bat = bat_hist.iloc[-1]


    bat_input = pd.DataFrame([{
        "batter": batter,
        "venue": venue,
        "team1": team1,
        "team2": team2,
        "runs_last_5_avg": last_bat["runs_last_5_avg"],
        "runs_last_10_avg": last_bat["runs_last_10_avg"],
        "career_runs_avg": last_bat["career_runs_avg"],
        "career_sr": last_bat["career_sr"],
        "venue_runs_avg": last_bat["venue_runs_avg"],
        "pvt_runs_avg": last_bat["pvt_runs_avg"],
        "pvp_runs_avg": last_bat["pvp_runs_avg"]
    }])


    pred_runs = round(
        runs_model.predict(bat_input)[0],
        2
    )


    confidence = (
        "High" if pred_runs > 40
        else "Medium" if pred_runs > 25
        else "Low"
    )


    # ======================================
    # BOWLER SECTION
    # ======================================

    bowl_hist = bowler_df[
        bowler_df["bowler"] == bowler
    ].sort_values("date")

    last_bowl = bowl_hist.iloc[-1]


    bowl_dict = {
        "overs": last_bowl["overs"],
        "runs": last_bowl["runs"],
        "economy": last_bowl["economy"],
        "wickets_last_5": last_bowl["wickets_last_5"],
        "career_wickets_avg": last_bowl["career_wickets_avg"],
        "venue_wickets_avg": last_bowl["venue_wickets_avg"]
    }


    wkt_features = wickets_model.get_booster().feature_names


    bowl_input = pd.DataFrame(
        [[bowl_dict[f] for f in wkt_features]],
        columns=wkt_features
    )


    pred_wkts = round(
        wickets_model.predict(bowl_input)[0],
        2
    )


    # ======================================
    # KPI CARDS
    # ======================================

    c1, c2 = st.columns(2)

    with c1:
        st.metric("🏏 Predicted Runs", pred_runs)
        st.write("Confidence:", confidence)

    with c2:
        st.metric("🎯 Predicted Wickets", pred_wkts)


    # ======================================
    # FORM CHARTS
    # ======================================

    st.subheader("📈 Player Form")

    colf1, colf2 = st.columns(2)

    with colf1:

        fig1 = px.line(
            bat_hist.tail(10),
            x="date",
            y="runs",
            markers=True,
            title=f"{batter} - Last 10 Matches"
        )

        st.plotly_chart(fig1, use_container_width=True)


    with colf2:

        fig2 = px.line(
            bowl_hist.tail(10),
            x="date",
            y="wickets",
            markers=True,
            title=f"{bowler} - Last 10 Matches"
        )

        st.plotly_chart(fig2, use_container_width=True)


    # ======================================
    # ACTUAL vs PREDICTED
    # ======================================

    st.subheader("📊 Actual vs Predicted Runs")

    y_true = bat_hist["target_next_runs"].tail(20)

    y_pred = runs_model.predict(
        bat_hist[[
            "batter","venue","team1","team2",
            "runs_last_5_avg","runs_last_10_avg",
            "career_runs_avg","career_sr",
            "venue_runs_avg","pvt_runs_avg","pvp_runs_avg"
        ]].tail(20)
    )


    comp_df = pd.DataFrame({
        "Actual": y_true,
        "Predicted": y_pred
    })


    fig3 = px.scatter(
        comp_df,
        x="Actual",
        y="Predicted"
    )

    fig3.update_layout(
        title="Actual vs Predicted Runs",
        xaxis_title="Actual Runs",
        yaxis_title="Predicted Runs"
    )

    st.plotly_chart(fig3, use_container_width=True)


    # ======================================
    # RESIDUAL PLOT
    # ======================================

    st.subheader("📉 Residual Analysis")

    residuals = y_true.values - y_pred


    res_df = pd.DataFrame({
        "Predicted": y_pred,
        "Residual": residuals
    })


    fig4 = px.scatter(
        res_df,
        x="Predicted",
        y="Residual"
    )

    st.plotly_chart(fig4, use_container_width=True)


    # ======================================
    # FEATURE IMPORTANCE (RUNS MODEL)
    # ======================================

    st.subheader("🧠 Feature Importance (Runs Model)")

    model = runs_model.named_steps["model"]
    preprocessor = runs_model.named_steps["prep"]

    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_

    imp_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    })

    imp_df = imp_df.sort_values(
        "Importance",
        ascending=False
    ).head(10)

    fig5 = px.bar(
        imp_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top Feature Importance (Runs)"
    )

    st.plotly_chart(fig5, use_container_width=True)


    # ======================================
    # SAMPLE TABLE
    # ======================================

    st.subheader("📋 Sample Prediction Summary")

    table_df = pd.DataFrame({
        "Player": [batter, bowler],
        "Opponent": [team2, team1],
        "Venue": [venue, venue],
        "Prediction": [pred_runs, pred_wkts],
        "Confidence": [confidence, "-"]
    })

    st.dataframe(table_df, use_container_width=True)