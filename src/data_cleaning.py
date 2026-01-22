import pandas as pd

from src.config import PROCESSED_DIR
from src.utils import save_csv
from src.data_ingest import load_raw_data


def clean_matches(matches: pd.DataFrame) -> pd.DataFrame:
    matches = matches.copy()
    matches.columns = [c.strip().lower().replace(" ", "_") for c in matches.columns]

    if "id" not in matches.columns:
        raise ValueError("matches.csv must contain 'id' column")

    if "date" in matches.columns:
        matches["date"] = pd.to_datetime(matches["date"], errors="coerce")

    keep_cols = [c for c in [
        "id", "season", "date", "venue", "team1", "team2", "winner"
    ] if c in matches.columns]

    matches = matches[keep_cols].drop_duplicates(subset=["id"])
    return matches


def clean_deliveries(deliveries: pd.DataFrame) -> pd.DataFrame:
    deliveries = deliveries.copy()
    deliveries.columns = [c.strip().lower().replace(" ", "_") for c in deliveries.columns]

    # Standardize match id column -> match_id
    if "match_id" in deliveries.columns:
        pass
    elif "id" in deliveries.columns:
        deliveries = deliveries.rename(columns={"id": "match_id"})
    else:
        raise ValueError("deliveries.csv must contain 'match_id' or 'id' column")

    # Standardize batter column
    if "batter" in deliveries.columns:
        pass
    elif "batsman" in deliveries.columns:
        deliveries = deliveries.rename(columns={"batsman": "batter"})
    else:
        raise ValueError("deliveries.csv must contain 'batter' or 'batsman' column")

    # Ensure ball column exists (for counting balls faced)
    if "ball" not in deliveries.columns:
        # create ball count per match_id
        deliveries["ball"] = 1

    # Convert numeric columns safely
    numeric_cols = ["batsman_runs", "extra_runs", "total_runs", "is_wicket"]
    for col in numeric_cols:
        if col in deliveries.columns:
            deliveries[col] = pd.to_numeric(deliveries[col], errors="coerce").fillna(0)

    return deliveries


def run_cleaning_pipeline():
    matches, deliveries = load_raw_data()

    matches_clean = clean_matches(matches)
    deliveries_clean = clean_deliveries(deliveries)

    save_csv(matches_clean, PROCESSED_DIR / "matches_clean.csv")
    save_csv(deliveries_clean, PROCESSED_DIR / "deliveries_clean.csv")

    return matches_clean, deliveries_clean


if __name__ == "__main__":
    run_cleaning_pipeline()
