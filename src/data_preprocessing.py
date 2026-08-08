# src/data_preprocessing.py

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42


def main():
    df = pd.read_csv("data/raw/diabetes.csv")

    train, test = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    output_dir = Path("data/interim")
    output_dir.mkdir(parents=True, exist_ok=True)

    train.to_csv(output_dir / "train.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)


if __name__ == "__main__":
    main()
    