'''
This is where training dataset and backtesting dataset are
splited
'''

import pandas as pd
from src.data.clean import load_data
from src.features.engineer import add_rolling_features
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss

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
    "2023/24"
])].copy()

backtest_df = df[df["Season"] == "2024/25"].copy()

missing = training_df[training_df[feature_columns].isna().any(axis=1)]
# print(missing[["MatchDate", "HomeTeam", "AwayTeam"] + feature_columns])

# print(training_df.shape)
training_df = training_df.dropna(subset=feature_columns)
# print(training_df.shape)

# printDataSet(backtest_df, training_df)


X_train = training_df[feature_columns]
y_train = training_df["FullTimeResult"]

X_test = backtest_df[feature_columns]
y_test = backtest_df["FullTimeResult"]

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train, y_train)
probabilities = model.predict_proba(X_test)

print("Class order:", model.classes_)
print(probabilities[:5])

y_pred = model.predict(X_test)

labels = ["H", "D", "A"]
cm = confusion_matrix(y_test, y_pred, labels = labels)
print(pd.DataFrame(cm, index=[f"true{l}" for l in labels], columns = [f"pred{l}" for l in labels ]))

print("Log loss: ", log_loss(y_test, probabilities, labels=model.classes_))

print("=============================")
print("Predicted results:")
print(pd.Series(y_pred).value_counts())

print("Actual results:")
print(y_test.value_counts())

print("Model accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))