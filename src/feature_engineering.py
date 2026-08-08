# src/feature_engineering.py

from pathlib import Path
import pandas as pd


TARGET = "quality"


def create_features(df):
    # Add domain-specific transformations here.
    # Keep this function deterministic.

    return df


def main():
    input_dir = Path("data/interim")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(input_dir / "train.csv")
    test = pd.read_csv(input_dir / "test.csv")

    train = create_features(train)
    test = create_features(test)

    train.to_csv(output_dir / "train.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)


if __name__ == "__main__":
    main()