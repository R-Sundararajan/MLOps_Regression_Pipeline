# src/data_ingestion.py

from pathlib import Path
from sklearn.datasets import load_diabetes


def main():
    output_path = Path("data/raw/diabetes.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_diabetes(as_frame=True)
    dataset.frame.to_csv(output_path, index=False)

    print(f"Saved dataset to {output_path}")


if __name__ == "__main__":
    main()
    