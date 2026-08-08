# src/model_building.py

from pathlib import Path

import joblib
import pandas as pd
import yaml

from sklearn.linear_model import LinearRegression


def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    target = params["data"]["target"]

    train = pd.read_csv("data/processed/train.csv")

    X = train.drop(columns=[target])
    y = train[target]

    model = LinearRegression()
    model.fit(X, y)

    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/model.pkl")

    print("Model trained successfully.")


if __name__ == "__main__":
    main()