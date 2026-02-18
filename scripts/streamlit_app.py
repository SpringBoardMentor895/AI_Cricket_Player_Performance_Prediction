import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression

# Page Configuration
st.set_page_config(
    page_title="Cricket Player Performance Prediction",
    layout="wide"
)

# Title Section
st.markdown(
    "<h1 style='text-align: center;'>CRICKET PLAYER PERFORMANCE PREDICTION</h1>",
    unsafe_allow_html=True
)

# Load Data and Model Section
@st.cache_data
def load_batter_data():
    return pd.read_csv("../data/cleaned/batter_dataset.csv")

@st.cache_data
def load_bowler_data():
    try:
        return pd.read_csv("../data/cleaned/bowler_dataset.csv")
    except:
        return None

@st.cache_data
def load_reference_data():
    try:
        return pd.read_csv("../data/ref/reference.csv")
    except:
        return None

@st.cache_resource
def load_batter_model():
    return joblib.load("../models/batsman_model.joblib")

@st.cache_resource
def load_bowler_model():
    try:
        return joblib.load("../models/bowler_model.joblib")
    except:
        return None

@st.cache_resource
def load_shap_explainer():
    try:
        return joblib.load("../models/shap_explainer.pkl")
    except:
        return None

# Load all data
batter_data = load_batter_data()
bowler_data = load_bowler_data()
reference_data = load_reference_data()
batter_model = load_batter_model()
bowler_model = load_bowler_model()
shap_explainer = load_shap_explainer()

# Feature columns for batsman
batter_feature_cols = [
    "rolling_avg_5",
    "venue_avg",
    "pvt_avg",
    "pvp_avg",
    "career_avg",
    "rolling_strike_rate_10"
]

# Feature columns for bowler
bowler_feature_cols = [
    "rolling_avg_wickets",
    "overs_bowled_last5",
    "venue_wicket_rate",
    "bowler_career_avg"
]

# Function to calculate confidence
def calculate_confidence(prediction, std_dev, data_points):
    """Calculate confidence level based on prediction variance and data availability"""
    if data_points == 0:
        return "N/A"
    elif data_points < 3:
        return "Low"
    elif std_dev < 10 and data_points >= 5:
        return "High"
    elif std_dev < 15:
        return "Medium"
    else:
        return "Low"

# ============================================================
# SIDEBAR INPUT PARAMETERS
# ============================================================

st.sidebar.header("Input Parameters")

# Dropdown Selections
players = sorted(batter_data["batter"].unique())
teams = sorted(batter_data["bowling_team"].unique())
venues = sorted(batter_data["venue"].unique())

player = st.sidebar.selectbox("Player", players)
opponent = st.sidebar.selectbox("Opponent Team", teams)
venue = st.sidebar.selectbox("Venue", venues)

predict_btn = st.sidebar.button("PREDICT PERFORMANCE")

# ============================================================
# PREDICTION DASHBOARD
# ============================================================

# Filter Selected Player Data
player_batter_data = batter_data[
    (batter_data["batter"] == player) &
    (batter_data["bowling_team"] == opponent) &
    (batter_data["venue"] == venue)
].copy()

# Get recent player data (last 5 matches)
recent_player_data = batter_data[batter_data["batter"] == player].tail(5)

# Check if player is also a bowler
player_bowler_data = None
if bowler_data is not None:
    player_bowler_data = bowler_data[
        (bowler_data["bowler"] == player) &
        (bowler_data["venue"] == venue)
    ].copy()

if predict_btn:

    # Layout columns for predictions
    col1, col2 = st.columns(2)

    # ------------------------------------------------
    # BATSMAN PREDICTION
    # ------------------------------------------------
    with col1:
        if player_batter_data.empty:
            st.markdown(
                f"""
                <div style="background:#e6f7ff;padding:25px;border-radius:12px;text-align:center">
                    <h4 style="color:#000000;">PREDICTED RUNS</h4>
                    <h1 style="color:#000000;">--</h1>
                    <p style="color:#000000;">Confidence: N/A</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            # Take latest record for prediction
            X_batter = player_batter_data[batter_feature_cols].iloc[-1:]
            predicted_runs = int(batter_model.predict(X_batter)[0])
            
            # Calculate confidence
            std_dev = player_batter_data["next_match_runs"].std()
            data_points = len(player_batter_data)
            confidence = calculate_confidence(predicted_runs, std_dev, data_points)
            
            st.markdown(
                f"""
                <div style="background:#e6f7ff;padding:25px;border-radius:12px;text-align:center">
                    <h4 style="color:#000000;">PREDICTED RUNS</h4>
                    <h1 style="color:#000000;">{predicted_runs}</h1>
                    <p style="color:#000000;">Confidence: {confidence}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    # ------------------------------------------------
    # BOWLER PREDICTION
    # ------------------------------------------------
    with col2:
        if player_bowler_data is None or player_bowler_data.empty or bowler_model is None:
            st.markdown(
                f"""
                <div style="background:#ffe6f0;padding:25px;border-radius:12px;text-align:center">
                    <h4 style="color:#000000;">PREDICTED WICKETS</h4>
                    <h1 style="color:#000000;">--</h1>
                    <p style="color:#000000;">Confidence: N/A</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            X_bowler = player_bowler_data[bowler_feature_cols].iloc[-1:]
            predicted_wickets = int(round(bowler_model.predict(X_bowler)[0]))
            
            # Calculate confidence
            std_dev = player_bowler_data["next_match_wicket"].std()
            data_points = len(player_bowler_data)
            confidence = calculate_confidence(predicted_wickets, std_dev, data_points)
            
            st.markdown(
                f"""
                <div style="background:#ffe6f0;padding:25px;border-radius:12px;text-align:center">
                    <h4 style="color:#000000;">PREDICTED WICKETS</h4>
                    <h1 style="color:#000000;">{predicted_wickets}</h1>
                    <p style="color:#000000;">Confidence: {confidence}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    # ------------------------------------------------
    # Player Form Chart + SHAP Feature Importance
    # ------------------------------------------------
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Player Form: Last 5 IPL Matches")

        if len(recent_player_data) > 0:
            fig, ax = plt.subplots(figsize=(6, 4))
            
            match_numbers = range(1, len(recent_player_data) + 1)
            runs = recent_player_data["next_match_runs"].values
            
            ax.plot(match_numbers, runs, marker="o", color="#1f77b4", linewidth=2)
            ax.set_xlabel("Match")
            ax.set_ylabel("Runs")
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            plt.close()
        else:
            st.info("No recent match data available")

    with col4:
        st.subheader("SHAP Feature Importance for this Match")

        if not player_batter_data.empty and shap_explainer is not None:
            try:
                X_batter = player_batter_data[batter_feature_cols].iloc[-1:]
                shap_values = shap_explainer.shap_values(X_batter)
                
                # Create horizontal bar chart for SHAP values
                if isinstance(shap_values, np.ndarray):
                    shap_vals = shap_values[0] if shap_values.ndim > 1 else shap_values
                else:
                    shap_vals = shap_values
                
                # Create DataFrame for plotting
                shap_df = pd.DataFrame({
                    'feature': batter_feature_cols,
                    'shap_value': shap_vals
                })
                shap_df = shap_df.reindex(shap_df['shap_value'].abs().sort_values().index)
                
                fig, ax = plt.subplots(figsize=(6, 4))
                colors = ['red' if x < 0 else 'blue' for x in shap_df['shap_value']]
                ax.barh(shap_df['feature'], shap_df['shap_value'], color=colors)
                ax.set_xlabel('SHAP Value')
                ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                plt.tight_layout()
                
                st.pyplot(fig)
                plt.close()
            except Exception as e:
                st.info("SHAP explanation not available")
        else:
            st.info("SHAP explanation not available")

# ============================================================
# ANALYTICAL REPORT
# ============================================================

st.markdown("---")
st.markdown(
    "<h2 style='text-align: center;'>CRICKET PLAYER PERFORMANCE PREDICTION - ANALYTICAL REPORT</h2>",
    unsafe_allow_html=True
)
st.markdown("---")

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("Sample Predictions")
    
    st.markdown("Table 1: Sample predictions produced by the model (Confidence is an example derived metric)")
    
    # Create sample predictions
    sample_players = ["V. Kohli", "R. Sharma", "J. Bumrah"]
    sample_data = []
    
    for sample_player in sample_players:
        player_data = batter_data[batter_data["batter"] == sample_player]
        if len(player_data) > 0:
            latest = player_data.iloc[-1]
            X = player_data[batter_feature_cols].iloc[-1:] 
            pred = int(batter_model.predict(X)[0])
            std_dev = player_data["next_match_runs"].std()
            conf = calculate_confidence(pred, std_dev, len(player_data))
            
            sample_data.append({
                "Player": sample_player,
                "Opponent": latest["bowling_team"],
                "Venue": latest["venue"][:20],
                "Predicted Runs": pred,
                "Confidence": conf
            })
        
        # Add bowler if exists
        if sample_player == "J. Bumrah" and bowler_data is not None:
            bowler_player_data = bowler_data[bowler_data["bowler"] == sample_player]
            if len(bowler_player_data) > 0 and bowler_model is not None:
                latest = bowler_player_data.iloc[-1]
                X_w = bowler_player_data[bowler_feature_cols].iloc[-1:]
                pred_w = int(round(bowler_model.predict(X_w)[0]))
                std_dev = bowler_player_data["next_match_wicket"].std()
                conf = calculate_confidence(pred_w, std_dev, len(bowler_player_data))
                
                sample_data[-1] = {
                    "Player": sample_player,
                    "Opponent": latest["bowling_team"],
                    "Venue": latest["venue"][:20],
                    "Predicted Runs": f"{pred_w} wickets",
                    "Confidence": conf
                }
    
    if sample_data:
        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df, use_container_width=True, hide_index=True)

with col_right:
    # --------------------------------------------------------
    # Performance Charts for Selected Player
    # --------------------------------------------------------
    
    # Use the selected player from sidebar
    analysis_player = player
    
    # Filter data for analysis player
    analysis_data = batter_data[batter_data["batter"] == analysis_player].tail(50)
    
    if len(analysis_data) > 0:
        st.subheader(f"{analysis_player} - Last 10 Matches")
        
        recent_10 = analysis_data.tail(10)
        
        fig, ax = plt.subplots(figsize=(6, 3))
        match_nums = range(1, len(recent_10) + 1)
        ax.plot(match_nums, recent_10["next_match_runs"].values, marker="o", linewidth=2)
        ax.set_xlabel("Match")
        ax.set_ylabel("Runs")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()
    
    # Second row with 2 columns
    subcol1, subcol2 = st.columns(2)
    
    with subcol1:
        st.subheader("Actual vs Predicted Runs (Scatter Plot)")
        
        # Generate predictions for test set
        if len(analysis_data) > 10:
            X_analysis = analysis_data[batter_feature_cols]
            y_actual = analysis_data["next_match_runs"]
            y_pred = batter_model.predict(X_analysis)
            
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(y_actual, y_pred, alpha=0.6, s=30)
            
            # Add regression line
            lr = LinearRegression()
            lr.fit(y_actual.values.reshape(-1, 1), y_pred)
            x_line = np.array([y_actual.min(), y_actual.max()])
            y_line = lr.predict(x_line.reshape(-1, 1))
            ax.plot(x_line, y_line, color='red', linewidth=2, label='Trend')
            
            ax.set_xlabel("Actual Runs")
            ax.set_ylabel("Predicted Runs")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
    
    with subcol2:
        st.subheader("SHAP Feature Importance Chart")
        
        if shap_explainer is not None and len(analysis_data) > 0:
            try:
                # Calculate SHAP values for sample
                X_sample = analysis_data[batter_feature_cols].head(min(100, len(analysis_data)))
                shap_values = shap_explainer.shap_values(X_sample)
                
                # Calculate mean absolute SHAP values
                if isinstance(shap_values, np.ndarray):
                    mean_shap = np.abs(shap_values).mean(axis=0)
                else:
                    mean_shap = np.abs(shap_values).mean(axis=0)
                
                shap_importance = pd.DataFrame({
                    'feature': batter_feature_cols,
                    'importance': mean_shap
                }).sort_values('importance')
                
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.barh(shap_importance['feature'], shap_importance['importance'], color='steelblue')
                ax.set_xlabel('Mean |SHAP Value|')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            except:
                st.info("SHAP analysis not available")

# --------------------------------------------------------
# Bottom Section: SHAP Force Plot + Residual Plot
# --------------------------------------------------------
st.markdown("---")

bottom_col1, bottom_col2 = st.columns(2)

with bottom_col1:
    st.subheader(f"{analysis_player} - Prediction Explanation (SHAP Force Plot)")
    
    if shap_explainer is not None and len(analysis_data) > 0:
        try:
            X_latest = analysis_data[batter_feature_cols].iloc[-1:]
            shap_values = shap_explainer.shap_values(X_latest)
            
            # Create waterfall plot
            if isinstance(shap_values, np.ndarray):
                shap_vals = shap_values[0] if shap_values.ndim > 1 else shap_values
                base_value = shap_explainer.expected_value
            else:
                shap_vals = shap_values[0]
                base_value = shap_explainer.expected_value
            
            explanation = shap.Explanation(
                values=shap_vals,
                base_values=base_value,
                data=X_latest.iloc[0].values,
                feature_names=batter_feature_cols
            )
            
            fig = plt.figure(figsize=(8, 4))
            shap.plots.waterfall(explanation, show=False)
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.info("SHAP force plot not available")

with bottom_col2:
    st.subheader("Residuals vs Predicted Values (Residual Plot)")
    
    if len(analysis_data) > 10:
        X_analysis = analysis_data[batter_feature_cols]
        y_actual = analysis_data["next_match_runs"]
        y_pred = batter_model.predict(X_analysis)
        residuals = y_actual - y_pred
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(y_pred, residuals, alpha=0.6, s=30)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel("Predicted Values")
        ax.set_ylabel("Residuals")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()
