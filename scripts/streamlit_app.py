import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------
# Page Config
# ---------------------------------------------------
st.set_page_config(
    page_title="Cricket Player Performance Prediction",
    layout="wide"
)

st.markdown(
    "<h1 style='text-align:center;'>CRICKET PLAYER PERFORMANCE PREDICTION</h1>",
    unsafe_allow_html=True
)

# ---------------------------------------------------
# Load Data & Model
# ---------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("../data/processed/dataset.csv")

@st.cache_resource
def load_model():
    return joblib.load("xgb_model.joblib")

data = load_data()
model = load_model()

# IMPORTANT → must match training exactly
feature_cols = [
    "rolling_avg_5",
    "venue_avg",
    "pvt_avg",
    "career_avg"
]

# ---------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "",
    ["Prediction Dashboard", "Analytical Report"]
)

# ===================================================
# ===================================================
#                 PREDICTION PAGE
# ===================================================
# ===================================================

st.sidebar.header("Input Parameters")

players = sorted(data["batter"].unique())
teams = sorted(data["bowling_team"].unique())
venues = sorted(data["venue"].unique())

selected_player = st.sidebar.selectbox("Player", players)
selected_team = st.sidebar.selectbox("Opponent Team", teams)
selected_venue = st.sidebar.selectbox("Venue", venues)

predict_btn = st.sidebar.button("PREDICT PERFORMANCE")
if page == "Prediction Dashboard":
    player_data = data[
        (data["batter"] == selected_player) &
        (data["bowling_team"] == selected_team) &
        (data["venue"] == selected_venue)
    ]

    recent_data = data[data["batter"] == selected_player].tail(5)

    if predict_btn:

        if player_data.empty:
            st.warning("No historical data found for this selection.")
        else:
            X = player_data[feature_cols].iloc[-1:]
            predicted_runs = int(model.predict(X)[0])

            # ---------------- TOP CARDS ----------------

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(
                    f"""
                    <div style="background:#dbe9f4;padding:30px;border-radius:15px;text-align:center">
                        <h3>PREDICTED RUNS</h3>
                        <h1>{predicted_runs}</h1>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col2:
                st.markdown(
                    """
                    <div style="background:#f4f4f4;padding:30px;border-radius:15px;text-align:center">
                        <h3>PREDICTED WICKETS</h3>
                        <h1>--</h1>
                        <p>Confidence: N/A</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col3:
                std_dev = player_data["next_match_runs"].std()
                confidence = "High" if std_dev < 10 else "Medium"
                st.markdown(
                    f"""
                    <div style="background:#e6f4ea;padding:30px;border-radius:15px;text-align:center">
                        <h3>CONFIDENCE</h3>
                        <h2>{confidence}</h2>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # ---------------- BOTTOM SECTION ----------------

            col4, col5 = st.columns(2)

            # Player Form Chart
            with col4:
                st.subheader("Player Form: Last 5 IPL Matches")

                fig, ax = plt.subplots()
                ax.plot(
                    range(1, len(recent_data) + 1),
                    recent_data["next_match_runs"],
                    marker="o"
                )
                ax.set_xlabel("Match")
                ax.set_ylabel("Runs")
                ax.grid(True)

                st.pyplot(fig)
                plt.close()

            # SHAP Waterfall
            with col5:
                st.subheader("SHAP Feature Importance for this Match")

                try:
                    explainer = shap.Explainer(model)
                    shap_values = explainer(X)

                    fig2 = plt.figure()
                    shap.plots.waterfall(shap_values[0], show=False)
                    st.pyplot(fig2)
                    plt.close()
                except:
                    st.info("SHAP explanation not available")

# ===================================================
# ===================================================
#               ANALYTICAL REPORT PAGE
# ===================================================
# ===================================================

elif page == "Analytical Report":

    st.header("CRICKET PLAYER PERFORMANCE PREDICTION - ANALYTICAL REPORT")
    st.markdown("---")

    col_left, col_right = st.columns([1, 2])

    # ---------------- Sample Predictions ----------------
    with col_left:

        st.subheader("Sample Predictions")

        sample_players = data["batter"].unique()[:3]
        sample_data = []

        for p in sample_players:
            player_df = data[data["batter"] == p]

            if len(player_df) > 0:
                X_sample = player_df[feature_cols].iloc[-1:]
                pred = int(model.predict(X_sample)[0])

                sample_data.append({
                    "Player": p,
                    "Opponent": player_df.iloc[-1]["bowling_team"],
                    "Venue": player_df.iloc[-1]["venue"],
                    "Predicted Runs": pred
                })

        st.dataframe(pd.DataFrame(sample_data), use_container_width=True)

    # ---------------- Player Analysis ----------------
    with col_right:

        analysis_player = selected_player

        analysis_data = data[data["batter"] == analysis_player].tail(50)

        if len(analysis_data) > 0:

            st.subheader(f"{analysis_player} - Last 10 Matches")

            recent_10 = analysis_data.tail(10)

            fig, ax = plt.subplots()
            ax.plot(
                range(1, len(recent_10) + 1),
                recent_10["next_match_runs"],
                marker="o"
            )
            ax.set_xlabel("Match")
            ax.set_ylabel("Runs")
            ax.grid(True)

            st.pyplot(fig)
            plt.close()

            # Scatter Plot
            st.subheader("Actual vs Predicted Runs")

            X_all = analysis_data[feature_cols]
            y_actual = analysis_data["next_match_runs"]
            y_pred = model.predict(X_all)

            fig2, ax2 = plt.subplots()
            ax2.scatter(y_actual, y_pred)

            lr = LinearRegression()
            lr.fit(y_actual.values.reshape(-1, 1), y_pred)

            x_line = np.array([y_actual.min(), y_actual.max()])
            y_line = lr.predict(x_line.reshape(-1, 1))

            ax2.plot(x_line, y_line, color="red")
            ax2.set_xlabel("Actual")
            ax2.set_ylabel("Predicted")
            ax2.grid(True)

            st.pyplot(fig2)
            plt.close()

            # SHAP Summary
            st.subheader("SHAP Feature Importance")

            try:
                explainer = shap.Explainer(model)
                shap_values = explainer(X_all)

                fig3 = plt.figure()
                shap.summary_plot(shap_values, X_all, show=False)
                st.pyplot(fig3)
                plt.close()
            except:
                st.info("SHAP not available")

            # Residual Plot
            st.subheader("Residuals vs Predicted Values")

            residuals = y_actual - y_pred

            fig4, ax4 = plt.subplots()
            ax4.scatter(y_pred, residuals)
            ax4.axhline(0, color="red", linestyle="--")
            ax4.set_xlabel("Predicted")
            ax4.set_ylabel("Residuals")
            ax4.grid(True)

            st.pyplot(fig4)
            plt.close()
