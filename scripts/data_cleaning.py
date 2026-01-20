import pandas as pd
from pathlib import Path

def load_data(project_dir: Path):
    data_dir = project_dir / "data" / "raw"
    matches = pd.read_csv(data_dir / "matches.csv")
    deliveries = pd.read_csv(data_dir / "deliveries.csv")
    return matches, deliveries

def clean_data(matches: pd.DataFrame, deliveries: pd.DataFrame):
    matches["date"] = pd.to_datetime(matches["date"], errors="coerce")

    matches.loc[(matches["city"].isna()) & (matches["venue"] == "Sharjah Cricket Stadium"), "city"] = "Sharjah"
    matches.loc[(matches["city"].isna()) & (matches["venue"] == "Dubai International Cricket Stadium"), "city"] = "Dubai"

    matches["season"] = matches["season"].replace({
        "2020/21": "2020",
        "2009/10": "2010",
        "2007/08": "2008"
    }).astype(str)

    team_map = {
        "Mumbai Indians": "Mumbai Indians",
        "Chennai Super Kings": "Chennai Super Kings",
        "Kolkata Knight Riders": "Kolkata Knight Riders",
        "Royal Challengers Bangalore": "Royal Challengers Bangalore",
        "Royal Challengers Bengaluru": "Royal Challengers Bangalore",
        "Rajasthan Royals": "Rajasthan Royals",
        "Kings XI Punjab": "Kings XI Punjab",
        "Punjab Kings": "Kings XI Punjab",
        "Sunrisers Hyderabad": "Sunrisers Hyderabad",
        "Deccan Chargers": "Sunrisers Hyderabad",
        "Delhi Capitals": "Delhi Capitals",
        "Delhi Daredevils": "Delhi Capitals",
        "Gujarat Titans": "Gujarat Titans",
        "Gujarat Lions": "Gujarat Titans",
        "Lucknow Super Giants": "Lucknow Super Giants",
        "Pune Warriors": "Pune Warriors",
        "Rising Pune Supergiant": "Pune Warriors",
        "Rising Pune Supergiants": "Pune Warriors",
        "Kochi Tuskers Kerala": "Kochi Tuskers Kerala"
    }

    for col in ["team1", "team2", "winner", "toss_winner"]:
        matches[col] = matches[col].map(team_map)

    for col in ["batting_team", "bowling_team"]:
        deliveries[col] = deliveries[col].map(team_map)

    ipl_df = deliveries.merge(
        matches[["id", "season", "venue", "city", "winner", "date"]],
        left_on="match_id",
        right_on="id",
        how="left"
    )

    return matches, deliveries, ipl_df

def save_cleaned(project_dir: Path, ipl_df: pd.DataFrame):
    clean_dir = project_dir / "data" / "cleaned"
    clean_dir.mkdir(parents=True, exist_ok=True)

    out_path = clean_dir / "ipl_cleaned.csv"
    ipl_df.to_csv(out_path, index=False)
    print("Saved cleaned dataset to:", out_path)

if __name__ == "__main__":
    PROJECT_DIR = Path(__file__).resolve().parents[1]

    matches, deliveries = load_data(PROJECT_DIR)
    matches, deliveries, ipl_df = clean_data(matches, deliveries)

    print("Matches:", matches.shape)
    print("Deliveries:", deliveries.shape)
    print("IPL merged:", ipl_df.shape)

    save_cleaned(PROJECT_DIR, ipl_df)
