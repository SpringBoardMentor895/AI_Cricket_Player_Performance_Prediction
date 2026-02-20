import streamlit as st
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
import pandas as pd


#######_________page config__________________#####
st.set_page_config(
    page_title="IPL Player Performance Prediction",
    layout="wide"
)
st.markdown("""
<div style="
    display:flex;
    justify-content:center;
    align-items:center;
    gap:20px;
    padding:15px 0;
">
    <span style="font-size:60px;">🏏</span>
    <h1 style="
        font-size:45px;
        font-weight:800;
        background: linear-gradient(90deg,#4facfe,#00f2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin:0;
    ">
        IPL PLAYER PERFORMANCE PREDICTION
    </h1>
</div>
""", unsafe_allow_html=True)
#-------------------------------------button
st.markdown("""
<style>
.stButton > button {
    background-color: #1f77ff !important;
    color: white !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    padding: 0.6rem 1rem !important;
    border: none !important;
}
.stButton > button:hover {
    background-color: #155edb !important;
}
</style>
""", unsafe_allow_html=True)



#####_____________________load data& models__________#####
@st.cache_data
def load_data():
    return pd.read_csv("data/processed/dataset.csv")
@st.cache_data
def load_bowler_data():
    return pd.read_csv("data/processed/bowler_dataset.csv")

@st.cache_resource
def load_model():
    return joblib.load("xgb_model.joblib")
@st.cache_resource
def load_shap():
    return joblib.load("shap_explainer.pkl")

@st.cache_resource
def load_bowler_model():
    return joblib.load("xgb_bowler_model.joblib")

@st.cache_resource
def load_bowler_shap():
    return joblib.load("shap_explainer_bowler.pkl")


data = load_data()
bowler_data = load_bowler_data()
model = load_model()
shap_explainer = load_shap()
bowler_model = load_bowler_model()
bowler_shap_explainer = load_bowler_shap()
#__________________________________navigation button
page = st.sidebar.radio(
    "Navigation",
    ["Prediction Dashboard","Analytical Report"]
)

#______________________input parameters


# ---- Define dropdown lists once (safe) ----
# ------------ COMMON SETTINGS ------------
predicted_runs = "--"
predicted_wickets = "--"

# batting model features
feature_cols_bat = ["rolling_avg_10", "venue_avg", "pvt_avg","pvp_avg", "career_avg"]

# bowler model features (as in bowler_dataset.csv)
feature_cols_bowl = [
    "rolling_avg_wickets",
    "venue_avg_wickets",
    "opponent_wickets",
    "bowler_career_wickets_avg"
]

# ---- dropdown lists ----
players = sorted(set(data["batter"]).union(bowler_data["bowler"]))
teams   = sorted(data["bowling_team"].unique())
venues  = sorted(data["venue"].unique())

# --------------------------------- PREDICTION DASHBOARD ---------------------------------
if page == "Prediction Dashboard":

    left, right = st.columns([1, 3], gap="large", vertical_alignment="top")

    with left:
        st.markdown("### Input Parameters")
        player = st.selectbox("Player", players)
        opponent = st.selectbox("Opponent Team", teams)
        venue = st.selectbox("Venue", venues)
        predict_btn = st.button("PREDICT PERFORMANCE", use_container_width=True)

    with right:
        col2, col3 = st.columns([1.2, 1.2])

        # ---------- FILTER DATA ----------
        # batting data from dataset.csv
        player_data_bat = data[
            (data["batter"] == player) &
            (data["bowling_team"] == opponent) &
            (data["venue"] == venue)
        ].copy()

        # bowling data from bowler_dataset.csv
        player_data_bowl = bowler_data[
            (bowler_data["bowler"] == player) &
            (bowler_data["batting_team"] == opponent) &   # opponent team is batting_team in bowler dataset
            (bowler_data["venue"] == venue)
        ].copy()

        # recent form (batting)
        recent_player_data = data[data["batter"] == player].tail(5)

        # ---------- PREDICTIONS ----------
        if predict_btn:
            # Runs prediction (batter model)
            if player_data_bat.empty:
                st.info("No batting data found for this selection.")
            else:
                x_bat = player_data_bat[feature_cols_bat].iloc[-1:].copy()
                predicted_runs = int(round(float(model.predict(x_bat)[0])))

            # Wickets prediction (bowler model)
            if player_data_bowl.empty:
                # no bowling record for this combination
                pass
            else:
                # keep only required columns
                if all(c in player_data_bowl.columns for c in feature_cols_bowl):
                    x_bowl = player_data_bowl[feature_cols_bowl].iloc[-1:].copy()
                    pw = float(bowler_model.predict(x_bowl)[0])
                    pw = max(0.0, min(pw, 10.0))  # clamp between 0 and 10
                    predicted_wickets = int(round(pw))

        # -------------------------------------- PLAYER FORM + RUNS CARD --------------------------------------
        with col2:
            st.markdown(f"""
            <div style="background:#e8f1ff;padding:25px;border-radius:12px;text-align:center;
                        color:#0b1f44;box-shadow:0px 4px 12px rgba(0,0,0,0.12);">
              <h4 style="margin:0;color:#1f3b73;">PREDICTED RUNS</h4>
              <h1 style="margin:10px 0;color:#0b1f44;">{predicted_runs}</h1>
              <p style="margin:0;color:#4b5d7a;">Confidence: High</p>
            </div>
            """, unsafe_allow_html=True)

            st.subheader("Player Form: Last 5 IPL Matches")
            if recent_player_data.empty:
                st.info("No recent batting data available for this player.")
            else:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(
                    range(1, len(recent_player_data) + 1),
                    recent_player_data["next_match_runs"],
                    marker="o",
                    linewidth=2
                )
                ax.set_xlabel("Match")
                ax.set_ylabel("Runs")
                ax.set_title("Recent Performance Trend")
                ax.grid(True, linestyle="--", alpha=0.5)
                st.pyplot(fig)

        # ---------------------------------------- SHAP + WICKETS CARD ----------------------------------------
        with col3:
            st.markdown(f"""
            <div style="background:#f3f5f9;padding:25px;border-radius:12px;text-align:center;
                        color:#111;box-shadow:0px 4px 12px rgba(0,0,0,0.10);">
              <h4 style="margin:0;color:#333;">PREDICTED WICKETS</h4>
              <h1 style="margin:10px 0;color:#111;">{predicted_wickets}</h1>
              <p style="margin:0;color:#666;">Confidence: N/A</p>
            </div>
            """, unsafe_allow_html=True)

            st.subheader("SHAP Feature Importance for this Match")
            if predict_btn and not player_data_bat.empty:
                x_df = player_data_bat[feature_cols_bat].iloc[-1:].copy()
                sv = shap_explainer(x_df)
                vals = sv.values[0]
                names = x_df.columns

                imp = (
                    pd.Series(np.abs(vals), index=names)
                    .sort_values(ascending=False)
                    .head(6)
                    .sort_values()
                )

                ylabels = [c.replace("_", " ")[:18] for c in imp.index]
                fig, ax = plt.subplots(figsize=(6.8, 2.6), dpi=160)
                ax.barh(ylabels, imp.values)
                for s in ["top", "right", "left", "bottom"]:
                    ax.spines[s].set_visible(False)
                ax.tick_params(axis="y", labelsize=9)
                ax.tick_params(axis="x", bottom=False, labelbottom=False)
                fig.tight_layout(pad=0.4)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            else:
                st.info("Click PREDICT PERFORMANCE to view SHAP.")

# --------------------------------- ANALYTICAL REPORT ---------------------------------
elif page == "Analytical Report":

    # ---------- CENTERED REPORT HEADER ----------
    st.markdown("""
    <div style="text-align:center; margin-top:10px; margin-bottom:20px;">
        <h2 style="margin:0;">CRICKET PLAYER PERFORMANCE PREDICTION - ANALYTICAL REPORT</h2>
    </div>
    """, unsafe_allow_html=True)

    # keep content centered
    left, mid, right = st.columns([1, 6, 1])

    with mid:
        # ==============================
        # ROW 1: Sample Predictions + Scatter
        # ==============================
        r1c1, r1c2 = st.columns([1.2, 1])

        with r1c1:
            st.subheader("Sample Predictions (Batting Model)")
            sample_df = data.sample(min(10, len(data)), random_state=42)

            X_sample = sample_df[feature_cols_bat]
            y_actual = sample_df["next_match_runs"]
            y_pred = model.predict(X_sample)

            sample_df["Predicted Runs"] = np.round(y_pred, 2)

            st.dataframe(
                sample_df[["batter", "venue", "bowling_team", "next_match_runs", "Predicted Runs"]],
                use_container_width=True
            )

        with r1c2:
            st.subheader("Actual vs Predicted (Scatter)")
            fig, ax = plt.subplots(figsize=(4.3, 3.2))
            ax.scatter(y_actual, y_pred, alpha=0.7)

            max_val = max(float(y_actual.max()), float(np.max(y_pred)))
            ax.plot([0, max_val], [0, max_val], linestyle="--")

            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.tick_params(labelsize=8)
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("---")

        # ==============================
        # ROW 2: SHAP Global + Residuals
        # ==============================
        r2c1, r2c2 = st.columns(2)

        with r2c1:
            st.subheader("Global SHAP Feature Importance")
            try:
                # safety: sample up to 200 rows
                n_sample = min(200, len(data))
                shap_sample = data[feature_cols_bat].sample(n_sample, random_state=42)
                shap_values = shap_explainer.shap_values(shap_sample)

                plt.figure(figsize=(5.2, 3.2))
                shap.summary_plot(
                    shap_values,
                    shap_sample,
                    feature_names=feature_cols_bat,
                    show=False
                )
                st.pyplot(plt.gcf())
                plt.clf()
                plt.close()
            except Exception as e:
                st.warning(f"SHAP failed: {e}")

        with r2c2:
            st.subheader("Residual Plot (Batting Model)")
            residuals = y_actual - y_pred

            fig, ax = plt.subplots(figsize=(4.3, 3.2))
            ax.scatter(y_pred, residuals, alpha=0.7)
            ax.axhline(0, linestyle="--")

            ax.set_xlabel("Predicted")
            ax.set_ylabel("Residuals")
            ax.tick_params(labelsize=8)
            st.pyplot(fig)
            plt.close(fig)
