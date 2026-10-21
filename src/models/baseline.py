'''
This is where training dataset and backtesting dataset are
splited
'''

import pandas as pd
from src.data.clean import load_data
from src.evaluation.evaluate import evaluate_model
from src.features.engineer import add_rolling_features
from sklearn.linear_model import LogisticRegression

df = add_rolling_features(load_data())

# HELPER FUNCTIONS
def printDataSet(backtest_df, training_df):
    print("Training dataset:")
    print(training_df.shape)
    print(training_df["Season"].value_counts().sort_index())

    print("=======================================")

    print("Backtest dataset:")
    print(backtest_df.shape)
    print(backtest_df['Season'].value_counts().sort_index())

    print("=======================================")

    print(backtest_df["FullTimeResult"].value_counts(normalize= True))

# MAIN SCRIPT


feature_columns = [
    "Home_AvgGoalsFor5",
    "Home_AvgGoalsAgainst5",
    "Away_AvgGoalsFor5",
    "Away_AvgGoalsAgainst5"
]


training_df = df[df["Season"].isin([
    "2020/21",
    "2021/22",
    "2022/23",
])].copy()

validation_df = df[df["Season"] == "2023/24"].copy()

test_df = df[df["Season"] == "2024/25"].copy()

missing = training_df[training_df[feature_columns].isna().any(axis=1)]
# print(missing[["MatchDate", "HomeTeam", "AwayTeam"] + feature_columns])

training_df = training_df.dropna(subset=feature_columns)
validation_df = validation_df.dropna(subset = feature_columns)
test_df = test_df.dropna(subset = feature_columns)

X_train = training_df[feature_columns]
y_train = training_df["FullTimeResult"]

X_validation = validation_df[feature_columns]
y_validation = validation_df["FullTimeResult"]

# printDataSet(backtest_df, training_df)

'''
Fitting the model using Logistic Regression
'''

model = LogisticRegression(max_iter=1500)

model.fit(X_train, y_train)

y_pred = model.predict(X_validation)
y_proba = model.predict_proba(X_validation)

evaluate_model(
    y_true = y_validation,
    y_pred = y_pred,
    y_proba= y_proba,
    classes = model.classes_,
    y_train = y_train
)


