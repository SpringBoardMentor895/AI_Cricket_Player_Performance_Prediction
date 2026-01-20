# AI Cricket Player Performance Prediction

This project predicts the next match performance of cricket players using machine learning, specifically XGBoost models for runs and wickets.

## Project Structure

- `data/`: Raw, cleaned, interim, and processed datasets
- `notebooks/`: Jupyter notebooks for EDA, feature engineering, and model training
- `scripts/`: Python scripts for data cleaning
- `src/`: Source code modules for data loading, aggregation, feature engineering, labeling, and splitting
- `artifacts/`: Trained model files (excluded from Git)
- `requirements.txt`: Python dependencies

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Run data cleaning: `python scripts/data_cleaning.py`
3. Execute notebooks in order: 01_EDA.ipynb, 02_FeatureEngineering.ipynb, 03_ModelTraining.ipynb

## Models

- `xgb_next_match_runs.pkl`: Predicts runs in next match
- `xgb_next_match_wickets.pkl`: Predicts wickets in next match

## Usage

Load a trained model and predict:

```python
import pickle
with open('artifacts/xgb_next_match_runs.pkl', 'rb') as f:
    model = pickle.load(f)
prediction = model.predict(features)
```

## License

MIT