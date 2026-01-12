
import pandas as pd
import numpy as np

# -----------------------------
# 1. Load Raw Data
# -----------------------------
matches = pd.read_csv("data/raw/matches.csv")
deliveries = pd.read_csv("data/raw/deliveries.csv")

print("Raw data loaded successfully")

# -----------------------------
# 2. Remove Duplicate Rows
# -----------------------------
matches.drop_duplicates(inplace=True)
deliveries.drop_duplicates(inplace=True)

# -----------------------------
# 3. Handle Missing Values
# -----------------------------

# ---- Matches dataset ----
num_cols_matches = matches.select_dtypes(include=["int64", "float64"]).columns
for col in num_cols_matches:
    matches[col] = matches[col].fillna(matches[col].median())

cat_cols_matches = matches.select_dtypes(include=["object"]).columns
for col in cat_cols_matches:
    if not matches[col].mode().empty:
        matches[col] = matches[col].fillna(matches[col].mode()[0])

# ---- Deliveries dataset ----
num_cols_deliveries = deliveries.select_dtypes(include=["int64", "float64"]).columns
for col in num_cols_deliveries:
    deliveries[col] = deliveries[col].fillna(deliveries[col].median())

cat_cols_deliveries = deliveries.select_dtypes(include=["object"]).columns
for col in cat_cols_deliveries:
    if not deliveries[col].mode().empty:
        deliveries[col] = deliveries[col].fillna(deliveries[col].mode()[0])

# -----------------------------
# 4. Data Type Standardization
# -----------------------------
if "date" in matches.columns:
    matches["date"] = pd.to_datetime(matches["date"], errors="coerce")

# -----------------------------
# 5. Remove Irrelevant / Redundant Columns
# -----------------------------
drop_cols_matches = ["umpire3"] if "umpire3" in matches.columns else []
matches.drop(columns=drop_cols_matches, inplace=True)

# -----------------------------
# 6. Save Cleaned Data
# -----------------------------
matches.to_csv("data/cleaned_matches.csv", index=False)
deliveries.to_csv("data/cleaned_deliveries.csv", index=False)

print("Data cleaning completed successfully!")
print("Cleaned files saved in data/ directory")
