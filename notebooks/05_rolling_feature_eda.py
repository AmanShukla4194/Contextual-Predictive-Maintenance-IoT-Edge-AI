import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import load_all
from src.features.rolling_features import add_rolling_stats
from config import SENSOR_COLS, ROLLING_WINDOWS

if __name__ == "__main__":
    data = load_all()
    telemetry = data["telemetry"]

    print("=== Rolling Feature EDA ===")
    df = add_rolling_stats(telemetry, SENSOR_COLS, ROLLING_WINDOWS)

    rolling_cols = [c for c in df.columns if any(f"_mean_{w}h" in c or f"_std_{w}h" in c for w in ROLLING_WINDOWS)]
    print(f"\nTotal rolling features generated: {len(rolling_cols)}")

    print("\n=== Rolling Feature Summary Statistics ===")
    print(df[rolling_cols].describe().T[["mean", "std", "min", "max"]].round(3))

    print("\n=== Null counts in rolling features ===")
    nulls = df[rolling_cols].isnull().sum()
    print(nulls[nulls > 0])

    print("\n=== Sample: volt_mean_3h and volt_std_24h by machineID (first 5 machines) ===")
    sample_cols = ["machineID", "datetime"] + [c for c in rolling_cols if "volt" in c][:4]
    top5 = df[df["machineID"].isin(df["machineID"].unique()[:5])]
    print(top5[sample_cols].head(20).to_string(index=False))
