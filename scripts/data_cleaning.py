import pandas as pd
import os
os.makedirs("../data/processed", exist_ok=True)
matches = pd.read_csv("../data/raw/matches.csv")
deliveries = pd.read_csv("../data/raw/deliveries.csv")
print("Raw data loaded successfully")
matches.columns = matches.columns.str.lower().str.replace(' ', '_')
deliveries.columns = deliveries.columns.str.lower().str.replace(' ', '_')
matches['date'] = pd.to_datetime(matches['date'], errors='coerce')
matches['city'] = matches['city'].fillna('Unknown')
matches['venue'] = matches['venue'].fillna('Unknown')
matches = matches.dropna(subset=['winner'])
matches['win_by_runs'] = matches['win_by_runs'].fillna(0)
matches['win_by_wickets'] = matches['win_by_wickets'].fillna(0)
deliveries = deliveries.fillna(0)
matches = matches.rename(columns={'id': 'match_id'})
ipl_data = deliveries.merge(
    matches[['match_id', 'date', 'venue', 'team1', 'team2']],
    on='match_id',
    how='left'
)
output_path = "../data/processed/ipl_cleaned.csv"
ipl_data.to_csv(output_path, index=False)
print(f"Cleaned data saved successfully at {output_path} ✅")



