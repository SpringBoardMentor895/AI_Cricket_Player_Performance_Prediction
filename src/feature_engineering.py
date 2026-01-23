import pandas as pd
from src.config import PROCESSED_DIR
from src.utils import save_csv


def _ensure_cols(df: pd.DataFrame, cols: list, name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def build_player_match_base(matches: pd.DataFrame, deliveries: pd.DataFrame) -> pd.DataFrame:
    """
    Ball-by-ball -> Batter-Match level dataset
    """
    _ensure_cols(deliveries, ["match_id", "batter", "batsman_runs", "ball"], "deliveries_clean.csv")
    _ensure_cols(matches, ["id", "date", "season", "venue", "team1", "team2", "winner"], "matches_clean.csv")

    # Batter-match aggregation
    batter_match = (
        deliveries.groupby(["match_id", "batter"], as_index=False)
        .agg(
            runs=("batsman_runs", "sum"),
            balls_faced=("ball", "count")
        )
    )

    matches_small = matches[["id", "date", "season", "venue", "team1", "team2", "winner"]].copy()
    matches_small["date"] = pd.to_datetime(matches_small["date"], errors="coerce")

    df = batter_match.merge(matches_small, left_on="match_id", right_on="id", how="left")
    df = df.drop(columns=["id"])
    df = df.sort_values(["batter", "date"]).reset_index(drop=True)

    return df


def add_form_and_career_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling averages + career avg (NO leakage)
    """
    df = df.copy()

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

    df[["runs_last_5_avg", "runs_last_10_avg", "career_runs_avg"]] = df[
        ["runs_last_5_avg", "runs_last_10_avg", "career_runs_avg"]
    ].fillna(0)

    return df


def add_venue_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Venue avg runs for batter (NO leakage)
    """
    df = df.copy()

    df["venue_runs_avg"] = (
        df.groupby(["batter", "venue"])["runs"]
        .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
    )

    df["venue_runs_avg"] = df["venue_runs_avg"].fillna(0)
    return df


def add_pvt_features(df: pd.DataFrame, deliveries: pd.DataFrame) -> pd.DataFrame:
    """
    PvT = Player vs Team (batter vs bowling_team) historical avg runs (NO leakage)
    """
    df = df.copy()

    # Always create column
    df["pvt_runs_avg"] = 0

    # If bowling_team missing, return safely
    if "bowling_team" not in deliveries.columns:
        return df

    pvt_match = (
        deliveries.groupby(["match_id", "batter", "bowling_team"], as_index=False)
        .agg(pvt_runs=("batsman_runs", "sum"))
    )

    pvt_match = pvt_match.merge(df[["match_id", "date"]].drop_duplicates(), on="match_id", how="left")
    pvt_match = pvt_match.sort_values(["batter", "bowling_team", "date"]).reset_index(drop=True)

    pvt_match["pvt_runs_avg"] = (
        pvt_match.groupby(["batter", "bowling_team"])["pvt_runs"]
        .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
    )
    pvt_match["pvt_runs_avg"] = pvt_match["pvt_runs_avg"].fillna(0)

    pvt_final = (
        pvt_match.groupby(["match_id", "batter"], as_index=False)
        .agg(pvt_runs_avg=("pvt_runs_avg", "mean"))
    )

    df = df.drop(columns=["pvt_runs_avg"], errors="ignore")
    df = df.merge(pvt_final, on=["match_id", "batter"], how="left")
    df["pvt_runs_avg"] = df["pvt_runs_avg"].fillna(0)

    return df


def add_pvp_features(df: pd.DataFrame, deliveries: pd.DataFrame) -> pd.DataFrame:
    """
    PvP = Player vs Player (batter vs bowler) historical avg runs (NO leakage)
    """
    df = df.copy()

    # Always create column
    df["pvp_runs_avg"] = 0

    # If bowler missing, return safely
    if "bowler" not in deliveries.columns:
        return df

    pvp_match = (
        deliveries.groupby(["match_id", "batter", "bowler"], as_index=False)
        .agg(pvp_runs=("batsman_runs", "sum"))
    )

    pvp_match = pvp_match.merge(df[["match_id", "date"]].drop_duplicates(), on="match_id", how="left")
    pvp_match = pvp_match.sort_values(["batter", "bowler", "date"]).reset_index(drop=True)

    pvp_match["pvp_runs_avg"] = (
        pvp_match.groupby(["batter", "bowler"])["pvp_runs"]
        .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
    )
    pvp_match["pvp_runs_avg"] = pvp_match["pvp_runs_avg"].fillna(0)

    pvp_final = (
        pvp_match.groupby(["match_id", "batter"], as_index=False)
        .agg(pvp_runs_avg=("pvp_runs_avg", "mean"))
    )

    df = df.drop(columns=["pvp_runs_avg"], errors="ignore")
    df = df.merge(pvp_final, on=["match_id", "batter"], how="left")
    df["pvp_runs_avg"] = df["pvp_runs_avg"].fillna(0)

    return df



def add_next_match_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label = runs in NEXT match (Infosys requirement)
    """
    df = df.copy()
    df = df.sort_values(["batter", "date"]).reset_index(drop=True)

    df["target_next_runs"] = df.groupby("batter")["runs"].shift(-1)

    # remove last match for each batter (no next match label)
    df = df.dropna(subset=["target_next_runs"]).reset_index(drop=True)
    return df


def run_feature_engineering():
    matches = pd.read_csv(PROCESSED_DIR / "matches_clean.csv")
    deliveries = pd.read_csv(PROCESSED_DIR / "deliveries_clean.csv")

    matches["date"] = pd.to_datetime(matches["date"], errors="coerce")

    df = build_player_match_base(matches, deliveries)
    df = add_form_and_career_features(df)
    df = add_venue_features(df)
    df = add_pvt_features(df, deliveries)
    df = add_pvp_features(df, deliveries)
    df = add_next_match_label(df)

    save_csv(df, PROCESSED_DIR / "dataset.csv")
    return df


if __name__ == "__main__":
    run_feature_engineering()
