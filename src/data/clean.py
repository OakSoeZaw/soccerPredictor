"""
This file is going to read the raw data
and clean it into the format that I can use to
train the model
"""

import pandas as pd

df = pd.read_csv("data/raw/epl_final.csv")

print("=====================================")
print(df['MatchDate'].head())
print(df['MatchDate'].tail())

print("=====================================")

df["MatchDate"] = pd.to_datetime(df['MatchDate'])
print(df['MatchDate'])