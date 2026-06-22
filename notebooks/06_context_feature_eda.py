import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import load_all
from src.features.context_features import time_since_maintenance, cumulative_error_count

if __name__ == "__main__":
    data = load_all()
    telemetry = data["telemetry"]
    maintenance = data["maintenance"]
    errors = data["errors"]

    print("=== Context Feature EDA ===")

    df = time_since_maintenance(telemetry, maintenance)
    df = cumulative_error_count(df, errors)

    print("\n--- hours_since_maintenance ---")
    print(df["hours_since_maintenance"].describe().round(2))
    print(f"Null count: {df['hours_since_maintenance'].isnull().sum()}")

    print("\n--- cumulative_error_count ---")
    print(df["cumulative_error_count"].describe().round(2))
    print(f"Machines with zero cumulative errors: {(df.groupby('machineID')['cumulative_error_count'].max() == 0).sum()}")

    print("\n--- Correlation with each other ---")
    ctx_cols = ["hours_since_maintenance", "cumulative_error_count"]
    print(df[ctx_cols].corr().round(3))

    print("\n--- Sample output (first 10 rows) ---")
    print(df[["machineID", "datetime"] + ctx_cols].head(10).to_string(index=False))