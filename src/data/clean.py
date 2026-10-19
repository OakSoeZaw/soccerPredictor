"""
This file is going to read the raw data
and clean it into the format that I can use to
train the model
"""

import pandas as pd

def load_data(): 
    df = pd.read_csv("data/raw/epl_final.csv")

    df["MatchDate"] = pd.to_datetime(df['MatchDate'])


    df = df.sort_values(["MatchDate", "HomeTeam"]).reset_index(drop = True)

    return df

'''
Reshaping the df so that it will be away and home games for previous
5 matches. Renaming the columns for easier to reading and manipulation
'''





