# src/model_evaluation.py

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def main():
    model = joblib.load("models/model.pkl")

    test = pd.read_csv("data/processed/test.csv")

    X = test.drop(columns=["target"])
    y = test["target"]

    predictions = model.predict(X)

    metrics = {
        "mae": mean_absolute_error(y, predictions),
        "mse": mean_squared_error(y, predictions),
        "rmse": mean_squared_error(y, predictions) ** 0.5,
        "r2": r2_score(y, predictions),
    }

    Path("reports").mkdir(exist_ok=True)

    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print(metrics)


if __name__ == "__main__":
    main()