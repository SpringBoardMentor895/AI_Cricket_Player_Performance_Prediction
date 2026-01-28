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
