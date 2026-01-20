import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
INTERIM_PATH = PROJECT_DIR / "data" / "interim" / "player_match.csv"
OUT_PATH = PROJECT_DIR / "data" / "processed" / "dataset.csv"

df = pd.read_csv(INTERIM_PATH)
print("Loaded player_match:", df.shape)

# Convert date
df["date"] = pd.to_datetime(df["date"], errors="coerce")
print("Missing dates:", df["date"].isna().sum())

# Rename columns to standard names (clean + consistent)
df = df.rename(columns={
    "batter": "player",
    "wickets_lost": "wickets"
})

# Sort for time-series shift
df = df.sort_values(["player", "date"])

# Create labels (next match stats for same player)
df["next_match_runs"] = df.groupby("player")["runs"].shift(-1)
df["next_match_wickets"] = df.groupby("player")["wickets"].shift(-1)

print("Missing next_match_runs:", df["next_match_runs"].isna().sum())
print("Missing next_match_wickets:", df["next_match_wickets"].isna().sum())

# Drop only label NaNs
df = df.dropna(subset=["next_match_runs", "next_match_wickets"])

# Convert labels to int
df["next_match_runs"] = df["next_match_runs"].astype(int)
df["next_match_wickets"] = df["next_match_wickets"].astype(int)

print("After labeling:", df.shape)

# Save dataset
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_PATH, index=False)

print("Saved dataset to:", OUT_PATH)
