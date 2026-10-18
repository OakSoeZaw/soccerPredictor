"""
This file is going to read the raw data
and clean it into the format that I can use to
train the model
"""

import pandas as pd

df = pd.read_csv("data/raw/epl_final.csv")

df["MatchDate"] = pd.to_datetime(df['MatchDate'])


df = df.sort_values(["MatchDate", "HomeTeam"]).reset_index(drop = True)


'''
Reshaping the df so that it will be away and home games for previous
5 matches. Renaming the columns for easier to reading and manipulation
'''

home = df[["MatchDate", "Season", "HomeTeam", "FullTimeHomeGoals", "FullTimeAwayGoals"]].copy()
home.columns = ["MatchDate", "Season", "Team", "GoalsFor", "GoalsAgainst"]
home["MatchID"] = df.index()
home["IsHome"] = 1

away = df[["MatchDate", "Season", "AwayTeam", "FullTimeAwayGoals", "FullTimeHomeGoals"]].copy()
away.columns = ["MatchDate", "Season", "Team", "GoalsFor", "GoalsAgainst"]
away["MatchID"] = df.index()
away["IsHome"] = 0

team_games = pd.concat([home, away]).sort_values(["Team", "MatchDate", "MatchID"]).reset_index(dopt = True)



training_df = df[df["Season"].isin([
    "2020/21",
    "2021/22",
    "2022/23",
    "2023/24"
])].copy()

print(training_df)

backtest_df = df[df["Season"] == "2024/25"].copy()


print("Training dataset:")
print(training_df.shape)
print(training_df["Season"].value_counts().sort_index())

print("=======================================")

print("Backtest dataset:")
print(backtest_df.shape)
print(backtest_df['Season'].value_counts().sort_index())

print("=======================================")