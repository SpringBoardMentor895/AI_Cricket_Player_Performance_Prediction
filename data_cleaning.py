import pandas as pd

# LOADING RAW DATA

matches = pd.read_csv("../data/raw/ipl_matches_2008_2024.csv")
balls = pd.read_csv("../data/raw/ipl_ball_by_ball_2008_2024.csv")

print("Raw datasets loaded")

# STANDARDIZING COLUMN NAMES

matches.columns = matches.columns.str.lower().str.strip()
balls.columns = balls.columns.str.lower().str.strip()

# DATE CONVERSION

matches['date'] = pd.to_datetime(matches['match_date'], errors='coerce')

# MISSING VALUE HANDLING

cat_cols_matches = ['venue', 'winner', 'team1', 'team2']
for col in cat_cols_matches:
    if col in matches.columns:
        matches[col] = matches[col].fillna('Unknown')

num_cols_balls = ['batsman_run', 'extras_run', 'total_run', 'is_wicket']
for col in num_cols_balls:
    if col in balls.columns:
        balls[col] = balls[col].fillna(0)

# Removing invalid rows
balls.dropna(subset=['batter', 'bowler'], inplace=True)

# STANDARDIZING TEAM NAMES
 
team_mapping = {
    'delhi daredevils': 'delhi capitals',
    'rising pune supergiant': 'rising pune supergiants',
    'royal challengers bangalore': 'royal challengers bengaluru'
}

for col in ['batting_team', 'bowling_team']:
    if col in balls.columns:
        balls[col] = balls[col].str.lower().replace(team_mapping)

for col in ['team1', 'team2', 'winner']:
    if col in matches.columns:
        matches[col] = matches[col].str.lower().replace(team_mapping)

# CLEAN VENUE NAMES

matches['venue'] = matches['venue'].str.strip().str.title()

# REMOVE DUPLICATES 

matches.drop_duplicates(inplace=True)
balls.drop_duplicates(inplace=True)

# MERGE DATASETS

ipl = balls.merge(
    matches[['id', 'date', 'venue', 'team1', 'team2']],
    left_on='id',
    right_on='id',
    how='left'
)

# Validate merge
missing_match_info = ipl['date'].isna().sum()
print(f"Missing match info after merge: {missing_match_info}")


print("Final dataset shape:", ipl.shape)
print("Final dataset columns:")
print(ipl.columns)

# CLEANED DATA

ipl.to_csv("../data/cleaned/ipl_cleaned.csv", index=False)

print("✅ Data cleaning completed successfully")