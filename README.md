Cricket Player Performance Prediction using IPL data.
Model: XGBoost
Features: Rolling averages, venue stats.
Run: streamlit run streamlit_app.py

Week 1–2: Data Acquisition & Exploratory Analysis
• Acquire and merge IPL datasets (ball-by-ball, match summaries).
• Inspect data schema, quality, and distributions.
• Clean data: handle missing values, normalize columns, ensure
date formats.
• EDA: distributions of runs, wickets, venue stats, team performance.
• Deliverable: Jupyter notebook (01_EDA.ipynb) with key
plots and initial findings; data_cleaning.py script.

Week 3–4: Feature Engineering & Preprocessing
• Aggregate ball-by-ball to player-match level.
• Engineer features: rolling averages (form), venue averages,
   opponent-specific stats (PvT, PvP), career stats.
• Create training labels (runs/wickets in the next match).
• Train-test split (time-series aware).
• Deliverable: Final feature-engineered dataset (dataset.csv),
   notebook (02_FeatureEngineering.ipynb), and saved preprocessor (feature_pipeline.pkl).

 Week 5–6: Model Development & Evaluation
• Establish baseline model (e.g., 10-match rolling average).
• Train models: Random Forest, XGBoost, LightGBM.
• Hyperparameter tuning: GridSearchCV or Optuna.
• Evaluation: RMSE, MAE, R2
. Analyze feature importance using SHAP.
• Deliverable: Trained model artifacts (xgb_model.joblib),
 notebook (03_ModelTraining.ipynb) with metrics and tuning logs.

Week 7–8: Dashboard, Deployment & Finalization
• Build Streamlit dashboard: inputs, prediction logic, and
  visualizations.
• Integrate model and preprocessor into the dashboard.
• Finalize documentation and clean repository.
• (Optional) Deploy app to Streamlit Cloud.
• Deliverable: Runnable Streamlit app (streamlit_app.py)
  and final README.md
