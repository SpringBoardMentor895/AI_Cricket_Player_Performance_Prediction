# ============================================================
# CRICKET PLAYER PERFORMANCE PREDICTION DASHBOARD + ANALYTICAL REPORT
# ============================================================

import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os
import numpy as np

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Cricket Player Performance Prediction",
    layout="wide"
)

st.title("CRICKET PLAYER PERFORMANCE PREDICTION")

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv("dataset.csv")

data = load_data()

# ============================================================
# LOAD MODEL + PIPELINE
# ============================================================

@st.cache_resource
def load_runs_model():
    return joblib.load("xgb_model.joblib")

@st.cache_resource
def load_pipeline():
    return joblib.load("feature_pipeline.pkl")

runs_model = load_runs_model()
pipeline = load_pipeline()

# Optional wickets model
wickets_model = None
if os.path.exists("model_wickets.pkl"):
    wickets_model = joblib.load("model_wickets.pkl")

# ============================================================
# SHAP EXPLAINER (for runs)
# ============================================================

explainer_runs = shap.TreeExplainer(runs_model)

# ============================================================
# PIPELINE INPUT COLUMNS
# ============================================================

pipeline_cols = [
    'batter',
    'venue',
    'opponent_team',
    'runs_avg_last_5',
    'runs_avg_last_10',
    'venue_avg_runs',
    'opponent_avg_runs',
    'career_avg_runs',
    'career_matches'
]

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

page = st.sidebar.radio(
    "Navigation",
    ["Prediction Dashboard", "Analytical Report"]
)

# ============================================================
# PREDICTION DASHBOARD
# ============================================================

if page == "Prediction Dashboard":

    st.sidebar.header("Input Parameters")

    players = sorted(data["batter"].unique())
    venues = sorted(data["venue"].unique())
    opponents = sorted(data["opponent_team"].unique())

    # Store player selection in session_state
    if 'pred_player_name' not in st.session_state:
        st.session_state['pred_player_name'] = players[0]

    pred_player_name = st.sidebar.selectbox(
        "Select Player", players, index=players.index(st.session_state['pred_player_name'])
    )
    st.session_state['pred_player_name'] = pred_player_name

    venue = st.sidebar.selectbox("Select Venue", venues)
    opponent = st.sidebar.selectbox("Select Opponent", opponents)

    predict_btn = st.sidebar.button("Predict Performance")

    # --------------------------
    # Get latest player record
    # --------------------------
    player_data = data[data["batter"] == pred_player_name]
    latest_record = player_data.iloc[-1:].copy()
    latest_record["venue"] = venue
    latest_record["opponent_team"] = opponent

    recent_data = player_data.tail(5)

    if predict_btn:

        # --------------------------
        # Prepare input
        # --------------------------
        X = latest_record[pipeline_cols]
        X_processed = pipeline.transform(X)

        # ====================================================
        # MAIN 2 COLUMNS
        # ====================================================
        col1, col2 = st.columns(2)

        # --------------------------
        # LEFT COLUMN: Predicted Runs + Player Form
        # --------------------------
        with col1:
            st.subheader("Predicted Runs")
            predicted_runs = int(runs_model.predict(X_processed)[0])
            st.markdown(
                f"<h1 style='color:black; font-weight:bold;'>{predicted_runs}</h1>", unsafe_allow_html=True
            )

            st.subheader("Player Form")
            fig_form, ax_form = plt.subplots()
            ax_form.plot(
                range(1, len(recent_data) + 1),
                recent_data["target_runs_next_match"],
                marker='o',
                color='blue'
            )
            ax_form.set_xlabel("Recent Matches")
            ax_form.set_ylabel("Runs")
            st.pyplot(fig_form)
            plt.clf()

        # --------------------------
        # RIGHT COLUMN: Predicted Wickets + SHAP
        # --------------------------
        with col2:
            st.subheader("Predicted Wickets")
            if wickets_model is not None:
                try:
                    wickets = wickets_model.predict(X_processed)[0]
                    st.markdown(
                        f"<h1 style='color:#4B0082; font-weight:bold;'>{round(wickets,2)}</h1>",
                        unsafe_allow_html=True
                    )
                except:
                    st.write("--")
            else:
                st.write("--")

            # SHAP Explanation
            st.subheader("SHAP Explanation Importance for this Match")
            try:
                numeric_features = [
                    'career_avg_runs',
                    'venue_avg_runs',
                    'opponent_avg_runs',
                    'runs_avg_last_10',
                    'runs_avg_last_5',
                    'career_matches'
                ]

                feature_names_pipeline = (
                    pipeline.get_feature_names_out()
                    if hasattr(pipeline, "get_feature_names_out") else None
                )

                if hasattr(X_processed, "toarray"):
                    X_dense = X_processed.toarray()
                else:
                    X_dense = X_processed

                shap_values_full = explainer_runs(X_dense)

                shap_dict = {}
                for f in numeric_features:
                    if feature_names_pipeline is not None:
                        idxs = [i for i, name in enumerate(feature_names_pipeline) if f in name]
                        shap_dict[f] = shap_values_full[0].values[idxs].sum()
                    else:
                        shap_dict[f] = shap_values_full[0].values[0]

                shap_numeric = shap.Explanation(
                    values=np.array(list(shap_dict.values())),
                    base_values=shap_values_full[0].base_values,
                    data=np.array([latest_record[f].values[0] for f in numeric_features]),
                    feature_names=numeric_features
                )

                fig_shap, ax_shap = plt.subplots(figsize=(8,5))
                shap.plots.waterfall(shap_numeric, show=False)
                st.pyplot(fig_shap)
                plt.clf()

            except Exception as e:
                st.write("SHAP plot not available:", e)

    else:
        st.info("Click Predict Performance")

# ============================================================
# ANALYTICAL REPORT
# ============================================================

if page == "Analytical Report":

    st.title("CRICKET PLAYER PERFORMANCE ANALYTICAL REPORT")

    # --------------------------
    # SAMPLE PREDICTIONS (Random sample)
    # --------------------------
    st.subheader("Sample Predictions")
    sample_df = data.sample(30)
    X_sample = sample_df[pipeline_cols]
    y_actual_sample = sample_df["target_runs_next_match"]
    y_pred_sample = runs_model.predict(pipeline.transform(X_sample))
    sample_df["Predicted Runs"] = y_pred_sample

    st.dataframe(sample_df[[
        "batter",
        "venue",
        "opponent_team",
        "target_runs_next_match",
        "Predicted Runs"
    ]])

    # --------------------------
    # DASHBOARD-SELECTED PLAYER ANALYTICS
    # --------------------------
    if 'pred_player_name' in st.session_state:
        player_name = st.session_state['pred_player_name']
    else:
        st.warning("Please select a player in Prediction Dashboard first")
        st.stop()

    player_data_full = data[data["batter"] == player_name]

    if not player_data_full.empty:

        # LAST 10 MATCHES FORM
        st.subheader(f"{player_name} - Last 10 Matches Form")
        player_last10 = player_data_full.tail(10)
        fig_form, ax_form = plt.subplots()
        ax_form.plot(
            range(1, len(player_last10)+1),
            player_last10["target_runs_next_match"],
            marker='o',
            color='blue'
        )
        ax_form.set_xlabel("Matches")
        ax_form.set_ylabel("Runs")
        ax_form.set_title(f"{player_name} - Last 10 Matches")
        st.pyplot(fig_form)
        plt.clf()

        # ACTUAL VS PREDICTED RUNS
        st.subheader(f"{player_name} - Actual vs Predicted Runs")
        X_player = player_data_full[pipeline_cols]
        y_actual_player = player_data_full["target_runs_next_match"]
        y_pred_player = runs_model.predict(pipeline.transform(X_player))

        fig_scatter, ax_scatter = plt.subplots()
        ax_scatter.scatter(y_actual_player, y_pred_player, color='green')
        ax_scatter.plot(
            [y_actual_player.min(), y_actual_player.max()],
            [y_actual_player.min(), y_actual_player.max()],
            linestyle="--",
            color='red'
        )
        ax_scatter.set_xlabel("Actual Runs")
        ax_scatter.set_ylabel("Predicted Runs")
        st.pyplot(fig_scatter)
        plt.clf()

        # PLAYER PREDICTION EXPLANATION (SHAP Waterfall)
        st.subheader(f"{player_name} - Prediction Explanation (SHAP Waterfall)")
        try:
            player_row = player_data_full.tail(1)
            X_row = player_row[pipeline_cols]
            X_processed_row = pipeline.transform(X_row)
            shap_values_row = explainer_runs(X_processed_row)

            feature_names_pipeline = (
                pipeline.get_feature_names_out() if hasattr(pipeline, "get_feature_names_out") else None
            )

            numeric_features = [
                'career_avg_runs',
                'venue_avg_runs',
                'opponent_avg_runs',
                'runs_avg_last_10',
                'runs_avg_last_5',
                'career_matches'
            ]
            numeric_features_existing = [f for f in numeric_features if f in X_row.columns]

            shap_dict = {}
            for f in numeric_features_existing:
                if feature_names_pipeline is not None:
                    idxs = [i for i, name in enumerate(feature_names_pipeline) if f in name]
                    shap_dict[f] = shap_values_row[0].values[idxs].sum()
                else:
                    shap_dict[f] = shap_values_row[0].values[0]

            shap_explanation = shap.Explanation(
                values=np.array(list(shap_dict.values())),
                base_values=shap_values_row[0].base_values,
                data=np.array([X_row[f].values[0] for f in numeric_features_existing]),
                feature_names=numeric_features_existing
            )

            fig_shap, ax_shap = plt.subplots(figsize=(8,5))
            shap.plots.waterfall(shap_explanation, show=False)
            st.pyplot(fig_shap)
            plt.clf()

            # SHAP FEATURE IMPORTANCE (Bar Chart)
            st.subheader(f"{player_name} - SHAP Feature Importance (Bar Chart)")
            shap_abs = np.abs(shap_explanation.values)
            feature_names = shap_explanation.feature_names

            fig_bar, ax_bar = plt.subplots(figsize=(8,5))
            ax_bar.barh(feature_names, shap_abs, color='orange')
            ax_bar.set_xlabel("Absolute SHAP Value")
            ax_bar.set_title("Feature Contribution for this Prediction")
            ax_bar.invert_yaxis()
            st.pyplot(fig_bar)
            plt.clf()

        except Exception as e:
            st.write("Player SHAP explanation not available:", e)

        # RESIDUALS vs PREDICTED
        st.subheader(f"{player_name} - Residuals vs Predicted Runs")
        residuals = y_actual_player - y_pred_player
        fig_resid, ax_resid = plt.subplots()
        ax_resid.scatter(y_pred_player, residuals, color='purple')
        ax_resid.axhline(0, linestyle="--", color='black')
        ax_resid.set_xlabel("Predicted Runs")
        ax_resid.set_ylabel("Residuals")
        st.pyplot(fig_resid)
        plt.clf()
