import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.data.loader import load_all
from config import SENSOR_COLS

if __name__ == "__main__":
    data = load_all()
    telemetry = data["telemetry"]

    print("=== Sensor Anomaly EDA ===")

    print("\n--- Global Z-Score Outlier Count (|z| > 3) per Sensor ---")
    for col in SENSOR_COLS:
        mean = telemetry[col].mean()
        std = telemetry[col].std()
        z = (telemetry[col] - mean) / (std + 1e-9)
        outlier_count = (z.abs() > 3).sum()
        pct = outlier_count / len(telemetry) * 100
        print(f"  {col}: {outlier_count} outliers ({pct:.2f}%)")

    print("\n--- Per-Machine Outlier Counts (top 10 most anomalous machines) ---")
    telemetry = telemetry.copy()
    for col in SENSOR_COLS:
        mean = telemetry.groupby("machineID")[col].transform("mean")
        std = telemetry.groupby("machineID")[col].transform("std")
        telemetry[f"{col}_z"] = (telemetry[col] - mean) / (std + 1e-9)

    z_cols = [f"{c}_z" for c in SENSOR_COLS]
    telemetry["max_z"] = telemetry[z_cols].abs().max(axis=1)
    anomalous = telemetry[telemetry["max_z"] > 3]
    top_machines = anomalous.groupby("machineID").size().sort_values(ascending=False).head(10)
    print(top_machines)

    print("\n--- IQR-Based Outlier Counts per Sensor ---")
    for col in SENSOR_COLS:
        q1 = telemetry[col].quantile(0.25)
        q3 = telemetry[col].quantile(0.75)
        iqr = q3 - q1
        outliers = ((telemetry[col] < q1 - 1.5 * iqr) | (telemetry[col] > q3 + 1.5 * iqr)).sum()
        print(f"  {col}: {outliers} IQR outliers")