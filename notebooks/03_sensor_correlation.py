import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.data.loader import load_all
from config import SENSOR_COLS

if __name__ == "__main__":
    data = load_all()
    telemetry = data["telemetry"]

    print("=== Sensor Correlation Matrix ===")
    corr = telemetry[SENSOR_COLS].corr().round(3)
    print(corr)

    print("\n=== Sensor Statistics per Machine Model ===")
    machines = data["machines"]
    merged = telemetry.merge(machines, on="machineID", how="left")
    print(merged.groupby("model")[SENSOR_COLS].mean().round(2))

    print("\n=== Sensor Statistics by Machine Age Group ===")
    merged["age_group"] = pd.cut(merged["age"], bins=[0, 5, 10, 15, 20],
                                  labels=["0-5", "6-10", "11-15", "16-20"])
    print(merged.groupby("age_group", observed=True)[SENSOR_COLS].mean().round(2))