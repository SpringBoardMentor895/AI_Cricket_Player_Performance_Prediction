import pandas as pd
from src.config import PROCESSED_DIR
from src.utils import save_csv


def make_batsman_match_dataset(matches: pd.DataFrame, deliveries: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a batter-match level dataset.
    Target: runs scored by batter in that match.
    """

    # Detect match id column
    match_col = None
    if "match_id" in deliveries.columns:
        match_col = "match_id"
    elif "id" in deliveries.columns:
        match_col = "id"
    else:
        raise ValueError("deliveries must contain 'match_id' or 'id' column")

    # Detect batter column
    batter_col = None
    if "batter" in deliveries.columns:
        batter_col = "batter"
    elif "batsman" in deliveries.columns:
        batter_col = "batsman"
    else:
        raise ValueError("deliveries must contain 'batter' or 'batsman' column")

    # Aggregate batter stats per match
    batter_match = (
        deliveries.groupby([match_col, batter_col], as_index=False)
        .agg(
            runs=("batsman_runs", "sum"),
            balls_faced=("ball", "count")
        )
        .rename(columns={match_col: "match_id", batter_col: "batter"})
    )

    # Merge match info
    matches_small = matches[["id", "date", "season", "venue", "team1", "team2", "winner"]].copy()
    matches_small["date"] = pd.to_datetime(matches_small["date"], errors="coerce")

    df = batter_match.merge(matches_small, left_on="match_id", right_on="id", how="left")
    df = df.drop(columns=["id"])

    # Sort for rolling features
    df = df.sort_values(["batter", "date"]).reset_index(drop=True)

    # Rolling features (no leakage)
    df["runs_last_5_avg"] = (
        df.groupby("batter")["runs"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    df["runs_last_10_avg"] = (
        df.groupby("batter")["runs"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    )

    df["career_runs_avg"] = (
        df.groupby("batter")["runs"]
        .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
    )

    df["venue_runs_avg"] = (
        df.groupby(["batter", "venue"])["runs"]
        .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
    )

    df = df.fillna({
        "runs_last_5_avg": 0,
        "runs_last_10_avg": 0,
        "career_runs_avg": 0,
        "venue_runs_avg": 0
    })

    return df


def run_feature_engineering():
    matches = pd.read_csv(PROCESSED_DIR / "matches_clean.csv")
    deliveries = pd.read_csv(PROCESSED_DIR / "deliveries_clean.csv")

    dataset = make_batsman_match_dataset(matches, deliveries)

    save_csv(dataset, PROCESSED_DIR / "batsman_match_features.csv")
    return dataset


if __name__ == "__main__":
    run_feature_engineering()
