import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import load_all
from src.features.lag_features import add_lag_features, add_diff_features
from config import SENSOR_COLS

if __name__ == "__main__":
    data = load_all()
    telemetry = data["telemetry"]

    print("=== Lag Feature EDA ===")
    df = add_lag_features(telemetry, SENSOR_COLS)
    df = add_diff_features(df, SENSOR_COLS)

    lag_cols = [c for c in df.columns if "_lag_" in c]
    diff_cols = [c for c in df.columns if "_diff_" in c]

    print(f"\nLag features: {len(lag_cols)}")
    print(f"Diff features: {len(diff_cols)}")

    print("\n--- Lag Feature Stats (first sensor) ---")
    print(df[lag_cols[:3]].describe().round(3))

    print("\n--- Diff Feature Stats (first sensor) ---")
    print(df[diff_cols[:3]].describe().round(3))

    print("\n--- Null counts (lag) ---")
    print(df[lag_cols].isnull().sum())

    print("\n--- Correlation: raw vs lag1 for volt ---")
    if "volt" in SENSOR_COLS and "volt_lag_1h" in df.columns:
        print(df[["volt", "volt_lag_1h"]].corr().round(3))



