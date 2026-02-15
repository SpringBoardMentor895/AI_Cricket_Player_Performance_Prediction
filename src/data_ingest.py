from src.config import RAW_DIR
from src.utils import safe_read_csv

def load_raw_data():
    """
    Expects these files inside data/raw/
    - matches.csv
    - deliveries.csv
    """
    matches_path = RAW_DIR / "matches.csv"
    deliveries_path = RAW_DIR / "deliveries.csv"

    matches = safe_read_csv(matches_path)
    deliveries = safe_read_csv(deliveries_path)

    return matches, deliveries

if __name__ == "__main__":
    load_raw_data()
