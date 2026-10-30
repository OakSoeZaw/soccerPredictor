from src.data.clean import load_data
from src.features.engineer import add_rolling_features
from src.features.elo import add_elo_features

df = add_rolling_features(load_data())
df, ratings = add_elo_features(df, return_ratings=True)

print(len(df))
print(df[["Home_Elo", "Away_Elo", "EloDiff"]].isna().sum())
print(df[["Home_Elo", "Away_Elo"]].head(10))

top = sorted(ratings.items(), key=lambda x : x[1], reverse=True)[:5]
print(top)