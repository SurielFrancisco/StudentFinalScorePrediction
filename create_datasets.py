import pandas as pd
from pathlib import Path

df = pd.read_csv("data/StudentsPerformance.csv")
output = Path("data")

for percentage in [25, 50, 100]:
    if percentage == 100:
        subset = df
    else:
        subset = df.sample(
            frac=percentage / 100,
            random_state=42
        )

    subset.to_csv(
        output / f"StudentsPerformance-{percentage}.csv",
        index=False
    )

    print(f"{percentage}%: {len(subset)} filas")