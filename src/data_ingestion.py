from pathlib import Path
import pandas as pd

URL = (
    "https://archive.ics.uci.edu/"
    "ml/machine-learning-databases/"
    "wine-quality/"
    "winequality-white.csv"
)

output = Path("data/raw/winequality-white.csv")
output.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(
    URL,
    sep=";"
)

df.to_csv(
    output,
    index=False
)

print(f"Saved dataset to {output}")
print(f"Shape: {df.shape}")