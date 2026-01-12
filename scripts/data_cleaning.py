import pandas as pd

ball = pd.read_csv("../data/raw/deliveries.csv")
matches = pd.read_csv("../data/raw/matches.csv")

ball.columns = ball.columns.str.lower().str.replace(" ","_")
matches.columns = matches.columns.str.lower().str.replace(" ","_")

matches['date'] = pd.to_datetime(matches['date'],errors ='coerce')

# seperate the numerical, categorical columns
num_cols = matches.select_dtypes(include=['int64','float64']).columns
obj_cols = matches.select_dtypes(include=['object']).columns


matches['winner'].fillna('No Result',inplace=True)
matches['player_of_match'].fillna('Unknown',inplace=True)
ball['player_dismissed'].fillna('Not Out',inplace=True)

ball = ball.drop_duplicates()
matches = matches.drop_duplicates()
# 260921


ipl = ball.merge(matches[['id','date','venue','team1','team2','winner','toss_winner']], left_on='match_id', right_on='id')
ipl.to_csv("../data/cleaned/ipl_cleaned.csv",index=False)

print("Data cleaning is completed!!")

ipl.drop(columns=['id'],inplace=True)  # to drop the duplicate column in the new merged dataframe (ipl)