import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap
import numpy as np

st.set_page_config(layout="wide")

# ======================================================
# LOAD DATA
# ======================================================
@st.cache_data
def load_data():
    runs_df = pd.read_csv("data/cleaned/player_runs_dataset.csv")
    wickets_df = pd.read_csv("data/cleaned/player_wicket_dataset.csv")
    return runs_df, wickets_df

runs_df, wickets_df = load_data()

# ======================================================
# LOAD MODELS (Pipeline)
# ======================================================
@st.cache_resource
def load_models():
    runs_model = joblib.load("models/xgb_runs_model.pkl")
    wicket_model = joblib.load("models/xgb_wicket_model.pkl")
    return runs_model, wicket_model

runs_model, wicket_model = load_models()

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.title("🏏 IPL Performance Dashboard")

menu = st.sidebar.radio(
    "Select Section",
    ["Prediction", "Analytical Report"]
)

batter = st.sidebar.selectbox("Select Batter", runs_df["batter"].unique())
venue = st.sidebar.selectbox("Select Venue", runs_df["venue"].unique())
bowling_team = st.sidebar.selectbox(
    "Select Bowling Team",
    runs_df["bowling_team"].unique()
)

# ======================================================
# ================== PREDICTION SECTION =================
# ======================================================
if menu == "Prediction":

    st.title("🏏 Player Performance Prediction")

    filtered = runs_df[
        (runs_df["batter"] == batter) &
        (runs_df["venue"] == venue) &
        (runs_df["bowling_team"] == bowling_team)
    ]

    if len(filtered) == 0:
        st.warning("No historical data available for this selection.")
    else:
        latest = filtered.sort_values("date").iloc[-1:]

        # Runs Prediction
        X_runs = latest[runs_model.feature_names_in_]
        predicted_runs = float(runs_model.predict(X_runs)[0])
        predicted_runs = round(predicted_runs, 2)

        # Wicket Prediction
        wicket_filtered = wickets_df[
            (wickets_df["batter"] == batter) &
            (wickets_df["venue"] == venue) &
            (wickets_df["bowling_team"] == bowling_team)
        ]

        if len(wicket_filtered) > 0:
            X_wicket = wicket_filtered.iloc[-1:][
                wicket_model.feature_names_in_
            ]
            predicted_wicket = float(wicket_model.predict(X_wicket)[0])
            predicted_wicket = round(predicted_wicket, 2)
        else:
            predicted_wicket = 0.00

        col1, col2 = st.columns(2)
        col1.metric("Predicted Runs", f"{predicted_runs}")
        col2.metric("Predicted Wickets", f"{predicted_wicket}")

        # =======================
        # Player Form – Last 5
        # =======================
        st.subheader("📊 Player Form – Last 5 Matches")

        last5 = runs_df[
            runs_df["batter"] == batter
        ].sort_values("date").tail(5)

        fig1, ax1 = plt.subplots()
        ax1.plot(last5["future_runs"].values)
        ax1.set_xlabel("Match")
        ax1.set_ylabel("Runs")
        st.pyplot(fig1)

        # =======================
        # SHAP Feature Importance
        # =======================
        st.subheader("🔍 SHAP Feature Importance")

        explainer = shap.Explainer(runs_model)
        shap_values = explainer(X_runs)

        fig2 = plt.figure()
        shap.plots.bar(shap_values, show=False)
        st.pyplot(fig2)


# ======================================================
# ================= ANALYTICAL REPORT ==================
# ======================================================
if menu == "Analytical Report":

    st.title("📈 Analytical Report")

    # =======================
    # Table 1: Sample Predictions
    # =======================
    st.subheader("📋 Table 1: Sample Predictions")

    sample = runs_df.sample(10)
    X_sample = sample[runs_model.feature_names_in_]

    sample["Predicted_Runs"] = runs_model.predict(X_sample)
    sample["Predicted_Runs"] = sample["Predicted_Runs"].round(2)

    st.dataframe(
        sample[["batter", "future_runs", "Predicted_Runs"]]
    )

    # =======================
    # SHAP Force Plot
    # =======================
    st.subheader("🔎 Prediction Explanation (SHAP Force Plot)")

    explainer = shap.Explainer(runs_model)
    shap_values = explainer(X_sample)

    fig3 = plt.figure()
    shap.plots.force(
        shap_values[0],
        matplotlib=True,
        show=False
    )
    st.pyplot(fig3)

    # =======================
    # Last 10 Matches
    # =======================
    st.subheader("📊 Last 10 Matches")

    last10 = runs_df[
        runs_df["batter"] == batter
    ].sort_values("date").tail(10)

    st.dataframe(last10)

    # =======================
    # Actual vs Predicted
    # =======================
    st.subheader("Actual vs Predicted Runs")

    X_all = runs_df[runs_model.feature_names_in_]
    y_actual = runs_df["future_runs"]
    y_pred = runs_model.predict(X_all)

    fig4, ax4 = plt.subplots()
    ax4.scatter(y_actual, y_pred)
    ax4.set_xlabel("Actual Runs")
    ax4.set_ylabel("Predicted Runs")
    st.pyplot(fig4)

    # =======================
    # SHAP Feature Importance
    # =======================
    st.subheader("SHAP Feature Importance Chart")

    fig5 = plt.figure()
    shap.plots.bar(shap_values, show=False)
    st.pyplot(fig5)

    # =======================
    # Residual Plot
    # =======================
    st.subheader("Residual vs Predicted Plot")

    residuals = y_actual - y_pred

    fig6, ax6 = plt.subplots()
    ax6.scatter(y_pred, residuals)
    ax6.axhline(0)
    ax6.set_xlabel("Predicted")
    ax6.set_ylabel("Residuals")
    st.pyplot(fig6)
