# =========================================================
# CRICKET PLAYER PERFORMANCE PREDICTION - STREAMLIT APP
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib as mpl

st.set_page_config(page_title="Cricket Performance Predictor", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #f9fafb; }
    section[data-testid="stSidebar"] { background-color: #ffffff; }
    div[data-baseweb="select"] * { background-color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# MATPLOTLIB THEME
# ---------------------------------------------------------
mpl.rcParams.update({
    "figure.facecolor": "#ffffff", "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#e5e7eb", "axes.labelcolor": "#6b7280",
    "xtick.color": "#9ca3af", "ytick.color": "#9ca3af",
    "text.color": "#374151", "grid.color": "#f3f4f6",
    "grid.linestyle": "--", "grid.linewidth": 0.6,
    "font.family": "sans-serif", "axes.titlesize": 11, "axes.labelsize": 10,
})

# ---------------------------------------------------------
# LOAD DATA & MODELS
# ---------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("../data/processed/dataset.csv")

@st.cache_resource
def load_runs_model():
    return joblib.load("rf_model.joblib")

@st.cache_resource
def load_runs_shap():
    return joblib.load("shap_explainer_batsmen.pkl")

@st.cache_resource
def load_wickets_model():
    return joblib.load("xgb_model.joblib")

@st.cache_resource
def load_wickets_shap():
    return joblib.load("shap_explainer_bowler.pkl")

data                   = load_data()
runs_model             = load_runs_model()
runs_shap_explainer    = load_runs_shap()
wickets_model          = load_wickets_model()
wickets_shap_explainer = load_wickets_shap()

feature_cols_runs    = ["rolling_avg_5", "venue_avg", "pvt_avg", "pvp_avg", "career_avg"]
feature_cols_wickets = ["rolling_wkts_5", "bowler_venue_avg", "pvt_avg", "pvp_avg", "bowler_career_avg", "bowler_vs_team_avg"]

# ---------------------------------------------------------
# PLAYER ROLE DETECTION
# ---------------------------------------------------------
batters_set = set(data["batter"].unique())
bowlers_set = set(data["bowler"].unique())
all_players = sorted(batters_set | bowlers_set)

def get_role(player):
    is_bat  = player in batters_set
    is_bowl = player in bowlers_set
    if is_bat and is_bowl:  return "allrounder"
    elif is_bat:            return "batter"
    elif is_bowl:           return "bowler"
    return None

role_labels = {"batter": "🏏 Batter", "bowler": "🎳 Bowler", "allrounder": "⭐ All-Rounder"}

# ---------------------------------------------------------
# SESSION STATE — keep player in sync across pages
# ---------------------------------------------------------
if "selected_player" not in st.session_state:
    st.session_state.selected_player = all_players[0]

# ---------------------------------------------------------
# CHART HELPERS
# ---------------------------------------------------------
def form_chart(ax, x, y, color, light, ylabel):
    ax.plot(list(x), list(y), marker="o", color=color, linewidth=2, markersize=6, markerfacecolor=light)
    ax.fill_between(list(x), list(y), alpha=0.08, color=color)
    ax.set_xlabel("Match", fontsize=9); ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

def scatter_plot(ax, xa, xp, color, edge, xlabel, ylabel, ref_max):
    ax.scatter(xa, xp, color=color, alpha=0.7, s=55, edgecolors=edge, linewidths=0.6)
    ax.plot([0, ref_max], [0, ref_max], linestyle="--", color="#d1d5db", linewidth=1)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(True); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

def resid_plot(ax, xp, resid, color, edge, xlabel):
    ax.scatter(xp, resid, color=color, alpha=0.7, s=55, edgecolors=edge, linewidths=0.6)
    ax.axhline(0, linestyle="--", color="#d1d5db", linewidth=1)
    ax.set_xlabel(xlabel); ax.set_ylabel("Residuals")
    ax.grid(True); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

def shap_waterfall(explainer, X_row, feat_cols):
    sv  = explainer(X_row)
    exp = shap.Explanation(
        values      = sv.values[0]      if hasattr(sv, "values")      else sv[0],
        base_values = sv.base_values[0] if hasattr(sv, "base_values") else explainer.expected_value,
        data        = X_row.iloc[0],
        feature_names = feat_cols
    )
    fig = plt.figure(figsize=(6, 3.8))
    shap.waterfall_plot(exp, show=False)
    return fig

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏏 Navigation")
    page = st.radio("", ["Prediction Dashboard", "Analytical Report"], label_visibility="collapsed")
    st.divider()

    # Single player selector — synced via session_state
    st.markdown("**Select Player**")
    selected_player = st.selectbox(
        "Player",
        all_players,
        index=all_players.index(st.session_state.selected_player),
        key="selected_player",       # automatically writes back to session_state
        label_visibility="collapsed"
    )

    player = st.session_state.selected_player
    role   = get_role(player)
    st.caption(f"Role: {role_labels.get(role, 'Unknown')}")

    teams  = sorted(data["bowling_team"].unique())
    venues = sorted(data["venue"].unique())

    if page == "Prediction Dashboard":
        st.divider()
        st.markdown("**Match Parameters**")
        opponent    = st.selectbox("Opponent Team", teams)
        venue       = st.selectbox("Venue", venues)
        predict_btn = st.button("Predict Performance", use_container_width=True, type="primary")
    else:
        # On Analytical Report page, show a note that the same player is used
        st.caption("ℹ️ Showing report for the player selected above.")
        predict_btn = False
        opponent    = None
        venue       = None


# =========================================================
# MAIN HEADING
# =========================================================
st.title("🏏 Cricket Player Performance Prediction")
st.divider()


# =========================================================
# PREDICTION DASHBOARD
# =========================================================
if page == "Prediction Dashboard":

    st.subheader(f"Prediction Dashboard — {player}")
    st.caption(f"{role_labels.get(role)} · vs {opponent} · {venue}" if opponent else "")

    if predict_btn:

        # ── BATTER ────────────────────────────────────────
        if role in ("batter", "allrounder"):
            bat_df = data[
                (data["batter"] == player) &
                (data["bowling_team"] == opponent) &
                (data["venue"] == venue)
            ].copy()
            recent = data[data["batter"] == player].tail(5)

            if bat_df.empty:
                st.warning("No batting data found for this player/opponent/venue combination.")
            else:
                pred_runs = int(runs_model.predict(bat_df[feature_cols_runs].iloc[-1:])[0])

                st.markdown("#### Batting Prediction")
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.metric("Predicted Runs", pred_runs)
                with c2:
                    fig, ax = plt.subplots(figsize=(6, 2.4))
                    form_chart(ax, range(1, len(recent)+1), recent["next_match_runs"], "#2563eb", "#93c5fd", "Runs")
                    ax.set_title("Last 5 Matches — Runs", fontsize=10)
                    fig.tight_layout(); st.pyplot(fig); plt.clf()

                st.markdown("**Runs — Feature Impact (SHAP)**")
                fig = shap_waterfall(runs_shap_explainer, bat_df[feature_cols_runs].iloc[-1:], feature_cols_runs)
                st.pyplot(fig); plt.clf()

        # ── BOWLER ────────────────────────────────────────
        if role in ("bowler", "allrounder"):
            if role == "allrounder":
                st.divider()

            bowl_df = data[
                (data["bowler"] == player) &
                (data["batting_team"] == opponent) &
                (data["venue"] == venue)
            ].copy()
            recent_b = data[data["bowler"] == player].tail(5)

            if bowl_df.empty:
                st.warning("No bowling data found for this player/opponent/venue combination.")
            else:
                pred_wkts = int(wickets_model.predict(bowl_df[feature_cols_wickets].iloc[-1:])[0])

                st.markdown("#### Bowling Prediction")
                c3, c4 = st.columns([1, 2])
                with c3:
                    st.metric("Predicted Wickets", pred_wkts)
                with c4:
                    fig, ax = plt.subplots(figsize=(6, 2.4))
                    form_chart(ax, range(1, len(recent_b)+1), recent_b["next_match_wicket"], "#10b981", "#6ee7b7", "Wickets")
                    ax.set_title("Last 5 Matches — Wickets", fontsize=10)
                    fig.tight_layout(); st.pyplot(fig); plt.clf()

                st.markdown("**Wickets — Feature Impact (SHAP)**")
                fig = shap_waterfall(wickets_shap_explainer, bowl_df[feature_cols_wickets].iloc[-1:], feature_cols_wickets)
                st.pyplot(fig); plt.clf()

    else:
        st.info(f"Select opponent and venue from the sidebar, then click **Predict Performance**.")


# =========================================================
# ANALYTICAL REPORT
# =========================================================
if page == "Analytical Report":

    st.subheader(f"Analytical Report — {player}")
    st.caption(role_labels.get(role, ""))

    if role is None:
        st.warning("No data found for this player.")
        st.stop()

    # Allrounders get tabs; pure batter/bowler renders directly
    if role == "allrounder":
        bat_tab, bowl_tab = st.tabs(["🏏 Batting", "🎳 Bowling"])
    else:
        bat_tab  = st if role == "batter" else None
        bowl_tab = st if role == "bowler" else None

    # ── BATTING SECTION ──────────────────────────────────
    if bat_tab is not None:
        ctx = bat_tab

        pdf = data[data["batter"] == player].copy()
        sdf = pdf.sample(min(30, len(pdf))).copy()
        y_a = sdf["next_match_runs"]
        y_p = runs_model.predict(sdf[feature_cols_runs])
        sdf["Predicted Runs"] = y_p

        ctx.markdown("#### Batting Analysis")
        ctx.markdown("**Sample Predictions**")
        with ctx.expander("Show Predictions Table", expanded=True):
            ctx.dataframe(
                sdf[["venue", "bowling_team", "next_match_runs", "Predicted Runs"]]
                .rename(columns={"venue": "Venue", "bowling_team": "Opponent", "next_match_runs": "Actual Runs"}),
                use_container_width=True
            )

        cc1, cc2 = ctx.columns(2)
        with cc1:
            ctx.markdown("**Actual vs Predicted Runs**")
            fig, ax = plt.subplots(figsize=(5, 4))
            scatter_plot(ax, y_a, y_p, "#3b82f6", "#1d4ed8", "Actual Runs", "Predicted Runs",
                         max(float(y_a.max()), float(y_p.max())) + 5)
            fig.tight_layout(); ctx.pyplot(fig); plt.clf()

        with cc2:
            ctx.markdown("**Residual Plot**")
            fig, ax = plt.subplots(figsize=(5, 4))
            resid_plot(ax, y_p, y_a - y_p, "#8b5cf6", "#7c3aed", "Predicted Runs")
            fig.tight_layout(); ctx.pyplot(fig); plt.clf()

        ctx.markdown("**Run Scoring Trend**")
        fig, ax = plt.subplots(figsize=(10, 3.5))
        form_chart(ax, range(len(pdf)), pdf["next_match_runs"], "#2563eb", "#93c5fd", "Runs")
        ax.set_xlabel("Match Index"); fig.tight_layout(); ctx.pyplot(fig); plt.clf()

        ctx.divider()
        ctx.markdown("**SHAP Feature Importance — Runs Model**")
        sv  = runs_shap_explainer(pdf[feature_cols_runs])
        fig = plt.figure(figsize=(8, 4))
        shap.summary_plot(sv, pdf[feature_cols_runs], show=False)
        ctx.pyplot(fig); plt.clf()

    # ── BOWLING SECTION ───────────────────────────────────
    if bowl_tab is not None:
        ctx = bowl_tab

        pdf = data[data["bowler"] == player].copy()
        sdf = pdf.sample(min(30, len(pdf))).copy()
        y_a = sdf["next_match_wicket"]
        y_p = wickets_model.predict(sdf[feature_cols_wickets])
        sdf["Predicted Wickets"] = y_p

        ctx.markdown("#### Bowling Analysis")
        ctx.markdown("**Sample Predictions**")
        with ctx.expander("Show Predictions Table", expanded=True):
            ctx.dataframe(
                sdf[["venue", "batting_team", "next_match_wicket", "Predicted Wickets"]]
                .rename(columns={"venue": "Venue", "batting_team": "Opponent", "next_match_wicket": "Actual Wickets"}),
                use_container_width=True
            )

        cc3, cc4 = ctx.columns(2)
        with cc3:
            ctx.markdown("**Actual vs Predicted Wickets**")
            fig, ax = plt.subplots(figsize=(5, 4))
            scatter_plot(ax, y_a, y_p, "#10b981", "#059669", "Actual Wickets", "Predicted Wickets", 6)
            fig.tight_layout(); ctx.pyplot(fig); plt.clf()

        with cc4:
            ctx.markdown("**Residual Plot**")
            fig, ax = plt.subplots(figsize=(5, 4))
            resid_plot(ax, y_p, y_a - y_p, "#f43f5e", "#be123c", "Predicted Wickets")
            fig.tight_layout(); ctx.pyplot(fig); plt.clf()

        ctx.markdown("**Wickets Trend**")
        fig, ax = plt.subplots(figsize=(10, 3.5))
        form_chart(ax, range(len(pdf)), pdf["next_match_wicket"], "#10b981", "#6ee7b7", "Wickets")
        ax.set_xlabel("Match Index"); fig.tight_layout(); ctx.pyplot(fig); plt.clf()

        ctx.divider()
        ctx.markdown("**SHAP Feature Importance — Wickets Model**")
        sv  = wickets_shap_explainer(pdf[feature_cols_wickets])
        fig = plt.figure(figsize=(8, 4))
        shap.summary_plot(sv, pdf[feature_cols_wickets], show=False)
        ctx.pyplot(fig); plt.clf()