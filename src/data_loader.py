import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

def load_data():
    raw_path = BASE_DIR / "data" / "raw"

    matches = pd.read_csv(raw_path / "matches.csv")
    deliveries = pd.read_csv(raw_path / "deliveries.csv")

    return matches, deliveries
