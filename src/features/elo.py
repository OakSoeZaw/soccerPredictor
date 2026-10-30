import pandas as pd

def add_elo_features(df, K= 20, home_adv = 80, shrink_fraction= 0.25, return_ratings=False):
    df = df.copy()
    assert df["MatchDate"].is_monotonic_increasing, "df must be sorted by MatchDate"
    assert (df.index == range(len(df))).all(), "df index must be 0..n-1"

    ratings = {}

    home_elos = [None] * len(df)
    away_elos = [None] * len(df)

    previous_season = None

    for date, day_matches in df.groupby("MatchDate", sort=True):

        season = day_matches["Season"].iloc[0]

        if previous_season is not None and season != previous_season:
            for team in ratings:
                ratings[team] = (
                    ratings[team]
                    + shrink_fraction * (1500 - ratings[team])
                )

        previous_season = season

        rating_changes = {}

        for idx, row in day_matches.iterrows():
            home = row["HomeTeam"]
            away = row["AwayTeam"]

            if home not in ratings:
                ratings[home] = 1500

            if away not in ratings:
                ratings[away] = 1500

            home_elo = ratings[home]
            away_elo = ratings[away]

            home_elos[idx] = home_elo
            away_elos[idx] = away_elo

            expected = 1 /(
                1 + 10 **(-(home_elo - away_elo + home_adv)/400)
            )

            result = row["FullTimeResult"]

            if result == "H":
                actual = 1
            elif result == "D":
                actual = 0.5
            elif result == "A":
                actual = 0
            else:
                raise ValueError(f"Unexpected match result: { result }")

            change = K * (actual - expected)

            rating_changes[home] = (
                rating_changes.get(home, 0) + change
            )
            rating_changes[away] = (
                rating_changes.get(away, 0) - change
            )

        for team, change in rating_changes.items():
            ratings[team] += change

    df["Home_Elo"] = home_elos
    df["Away_Elo"] = away_elos
    df["EloDiff"] = df["Home_Elo"] - df["Away_Elo"]

    if return_ratings:
        return df, ratings

    return df




