# =========================================================
# IPL EDA & DATA CLEANING (MERGED-FIRST PIPELINE)
# Equivalent to: notebooks/01_EDA.ipynb
# Project: IPL_EDA_PROJECT
# =========================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ---------------------------------------------------------
# 1. SETTINGS
# ---------------------------------------------------------
sns.set(style="whitegrid")
pd.set_option("display.max_columns", None)

# ---------------------------------------------------------
# 2. PATHS (SAFE, ABSOLUTE, ROBUST)
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DELIVERIES_PATH = os.path.join(
    BASE_DIR, "..", "data", "raw", "deliveries_updated_mens_ipl_upto_2024.csv"
)

MATCHES_PATH = os.path.join(
    BASE_DIR, "..", "data", "raw", "matches_updated_mens_ipl_upto_2024.csv"
)

CLEANED_DIR = os.path.join(BASE_DIR, "..", "data", "cleaned")
PLOTS_DIR = os.path.join(CLEANED_DIR, "plots")

os.makedirs(CLEANED_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Safety checks
if not os.path.exists(DELIVERIES_PATH):
    raise FileNotFoundError(f"Deliveries file not found: {DELIVERIES_PATH}")

if not os.path.exists(MATCHES_PATH):
    raise FileNotFoundError(f"Matches file not found: {MATCHES_PATH}")

# ---------------------------------------------------------
# 3. LOAD RAW DATA
# ---------------------------------------------------------
deliveries = pd.read_csv(DELIVERIES_PATH)
matches = pd.read_csv(MATCHES_PATH)

print("Raw datasets loaded")
print("Deliveries:", deliveries.shape)
print("Matches:", matches.shape)
print("-" * 60)

# ---------------------------------------------------------
# 4. MERGE DATASETS
# ---------------------------------------------------------
ipl_df = deliveries.merge(
    matches,
    on="matchId",
    how="inner"
)

print("Merged dataset shape:", ipl_df.shape)
print("-" * 60)

# ---------------------------------------------------------
# 5. FEATURE ENGINEERING
# ---------------------------------------------------------
# Total runs per ball (IMPORTANT)
ipl_df["total_runs"] = ipl_df["batsman_runs"] + ipl_df["extras"]

# ---------------------------------------------------------
# 6. DATA INSPECTION
# ---------------------------------------------------------
print("Merged Dataset Info:")
ipl_df.info()
print("-" * 60)

print("Missing Values (Merged Dataset):")
print(ipl_df.isnull().sum())
print("-" * 60)

# ---------------------------------------------------------
# 7. HANDLE MISSING VALUES
# ---------------------------------------------------------
num_cols = ipl_df.select_dtypes(include=["int64", "float64"]).columns
cat_cols = ipl_df.select_dtypes(include=["object"]).columns

id_cols = ["matchId", "over", "ball", "inning"]
event_cat_cols = ["player_dismissed", "dismissal_kind"]

# Numerical → median
for col in num_cols:
    if col not in id_cols and ipl_df[col].isnull().sum() > 0:
        ipl_df[col] = ipl_df[col].fillna(ipl_df[col].median())

# Categorical → mode / None
for col in cat_cols:
    if col in event_cat_cols:
        ipl_df[col] = ipl_df[col].fillna("None")
    else:
        if ipl_df[col].isnull().sum() > 0:
            ipl_df[col] = ipl_df[col].fillna(ipl_df[col].mode()[0])

# Future-proof dtype handling
ipl_df = ipl_df.infer_objects(copy=False)

print("Missing Values After Cleaning:")
print(ipl_df.isnull().sum())
print("-" * 60)

# ---------------------------------------------------------
# 8. MATCHES PER SEASON (MATCH-LEVEL)
# ---------------------------------------------------------
matches_per_season = (
    matches.groupby("season")["matchId"]
    .nunique()
    .reset_index(name="matches")
)

plt.figure(figsize=(10, 5))
sns.barplot(
    x="season",
    y="matches",
    data=matches_per_season,
    color="red"
)
plt.title("Matches Per Season")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/matches_per_season.png")
plt.close()

# ---------------------------------------------------------
# 9. RUNS PER BALL DISTRIBUTION
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.histplot(ipl_df["total_runs"], bins=10, kde=True, color="red")
plt.title("Runs Per Ball Distribution")
plt.xlabel("Runs")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/runs_per_ball.png")
plt.close()

# ---------------------------------------------------------
# 10. RUNS PER OVER DISTRIBUTION
# ---------------------------------------------------------
runs_per_over = (
    ipl_df.groupby(["matchId", "over"])["total_runs"]
    .sum()
)

plt.figure(figsize=(8, 5))
sns.histplot(runs_per_over, bins=30, color="red")
plt.title("Runs Per Over Distribution")
plt.xlabel("Runs per Over")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/runs_per_over.png")
plt.close()

# ---------------------------------------------------------
# 11. TOP 10 TEAMS BY TOTAL RUNS
# ---------------------------------------------------------
team_runs = (
    ipl_df.groupby("batting_team")["total_runs"]
    .sum()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 6))
sns.barplot(
    x=team_runs.head(10).values,
    y=team_runs.head(10).index,
    color="red"
)
plt.title("Top 10 Teams by Total Runs")
plt.xlabel("Runs")
plt.ylabel("Team")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/top_teams_by_runs.png")
plt.close()

# ---------------------------------------------------------
# 12. TOP 10 TEAMS BY WICKETS
# ---------------------------------------------------------
wickets_by_team = (
    ipl_df[ipl_df["dismissal_kind"] != "None"]
    .groupby("bowling_team")
    .size()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 6))
sns.barplot(
    x=wickets_by_team.head(10).values,
    y=wickets_by_team.head(10).index,
    color="red"
)
plt.title("Top 10 Teams by Wickets Taken")
plt.xlabel("Wickets")
plt.ylabel("Team")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/top_teams_by_wickets.png")
plt.close()

# ---------------------------------------------------------
# 13. SAVE CLEANED DATASET
# ---------------------------------------------------------
ipl_df.to_csv(
    os.path.join(CLEANED_DIR, "ipl_merged_cleaned.csv"),
    index=False
)

print("EDA + Cleaning Completed Successfully")
print("Cleaned dataset → data/cleaned/")
print("Plots → data/cleaned/plots/")
