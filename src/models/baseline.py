'''
This is where training dataset and backtesting dataset are
splited
'''


from src.data.clean import load_data
from src.features.engineer import add_rolling_features

df = add_rolling_features(load_data())

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

print(backtest_df["FullTimeResult"].value_counts(normalize= True))

