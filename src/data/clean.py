"""
This file is going to read the raw data
and clean it into the format that I can use to
train the model
"""

import pandas as pd

df = pd.read_csv("data/raw/epl_final.csv")

df["MatchDate"] = pd.to_datetime(df['MatchDate'])

training_df = df[df["Season"].isin([
    "2020/21",
    "2021/22",
    "2022/23",
    "2023/24"
])].copy()

backtest_df = df[df["Season"] == "2024/25"].copy()


print("Training dataset:")
print(training_df.shape)
print(training_df["Season"].value_counts().sort_index())

print("=======================================")

print("Backtest dataset:")
print(backtest_df.shape)
print(backtest_df['Season'].value_counts().sort_index())

print("=======================================")