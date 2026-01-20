import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
CLEAN_PATH = PROJECT_DIR / "data" / "cleaned" / "ipl_cleaned.csv"
OUT_PATH = PROJECT_DIR / "data" / "interim" / "player_match.csv"

df = pd.read_csv(CLEAN_PATH)

print("Loaded cleaned:", df.shape)

# ✅ Use correct column names
# match id column is: match_id
# batter column is: batter

player_match = df.groupby(["match_id", "batter"]).agg(
    runs=("batsman_runs", "sum"),
    balls_faced=("ball", "count"),
    fours=("batsman_runs", lambda x: (x == 4).sum()),
    sixes=("batsman_runs", lambda x: (x == 6).sum()),
    wickets_lost=("is_wicket", "sum"),
    season=("season", "first"),
    venue=("venue", "first"),
    city=("city", "first"),
    date=("date", "first"),
).reset_index()

# Strike Rate feature
player_match["strike_rate"] = (player_match["runs"] / player_match["balls_faced"]) * 100

# Save output
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
player_match.to_csv(OUT_PATH, index=False)

print("Saved player_match dataset:", player_match.shape)
print("Saved to:", OUT_PATH)
