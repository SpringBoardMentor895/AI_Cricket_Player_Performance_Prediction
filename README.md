Cricket Player Performance Prediction using IPL data.
Model: XGBoost
Features: Rolling averages, venue stats.

## Usage:
1. Run data cleaning: python data_cleaning.py
2. Run feature engineering: python feature_engineering.py
3. Train model and make predictions using the generated CSV files

## Output Files:
- dataset.csv: Complete feature-engineered dataset
- train_data.csv: Training set
- test_data.csv: Test set
- feature_pipeline.pkl: Preprocessing pipeline
