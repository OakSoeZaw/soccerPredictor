"""
Taking the dataframe and return it with the feature columns
added ex. team_game
"""

import pandas as pd


def add_rolling_features(df, window=5):
    df = df.copy()
    df["MatchID"] = df.index
    home = df[
        ["MatchDate", "Season", "HomeTeam", "FullTimeHomeGoals", "FullTimeAwayGoals", "HomeShotsOnTarget", "AwayShotsOnTarget"]
    ].copy()

    home.columns = ["MatchDate", "Season", "Team", "GoalsFor", "GoalsAgainst", "ShotsFor", "ShotsAgainst"]
    home["MatchID"] = df.index
    home["IsHome"] = 1

    away = df[
        ["MatchDate", "Season", "AwayTeam", "FullTimeAwayGoals", "FullTimeHomeGoals", "HomeShotsOnTarget", "AwayShotsOnTarget"]
    ].copy()
    away.columns = ["MatchDate", "Season", "Team", "GoalsFor", "GoalsAgainst", "ShotsFor", "ShotsAgainst"]
    away["MatchID"] = df.index
    away["IsHome"] = 0

    team_games = (
        pd.concat([home, away])
        .sort_values(["Team", "MatchDate", "MatchID"])
        .reset_index(drop=True)
    )

    team_games["AvgShotsFor"] = team_games.groupby("Team")["ShotsFor"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=10).mean()
    )

    team_games["AvgGoalsFor5"] = team_games.groupby("Team")["GoalsFor"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=10).mean()
    )

    team_games["AvgShotsAgainst"] = team_games.groupby("Team")["ShotsAgainst"].transform(
        lambda s: s.shift(1).rolling(10, min_periods = 10).mean()
    )

    team_games["AvgGoalsAgainst5"] = team_games.groupby("Team")[
        "GoalsAgainst"
    ].transform(lambda s: s.shift(1).rolling(10, min_periods=10).mean())

    home_feats = team_games[team_games["IsHome"] == 1][
        ["MatchID", "AvgGoalsFor5", "AvgGoalsAgainst5", "AvgShotsFor", "AvgShotsAgainst"]
    ].copy()

    home_feats = home_feats.rename(
        columns={
            "AvgGoalsFor5": "Home_AvgGoalsFor5",
            "AvgGoalsAgainst5": "Home_AvgGoalsAgainst5",
            "AvgShotsFor": "Home_AvgShotsFor",
            "AvgShotsAgainst": "Home_AvgShotsAgainst"
        }
    )

    away_feats = team_games[team_games["IsHome"] == 0][
        ["MatchID", "AvgGoalsFor5", "AvgGoalsAgainst5",  "AvgShotsFor", "AvgShotsAgainst"]
    ].copy()

    away_feats = away_feats.rename(
        columns={
            "AvgGoalsFor5": "Away_AvgGoalsFor5",
            "AvgGoalsAgainst5": "Away_AvgGoalsAgainst5",
            "AvgShotsFor": "Away_AvgShotsFor",
            "AvgShotsAgainst": "Away_AvgShotsAgainst"
        }
    )

    """
    Merging back the features of away and home games
    """

    df["MatchID"] = df.index
    df = df.merge(home_feats, on="MatchID", how="left")
    df = df.merge(away_feats, on="MatchID", how="left")

    return df
