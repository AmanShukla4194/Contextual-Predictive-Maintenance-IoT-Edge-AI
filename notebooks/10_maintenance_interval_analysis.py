import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.data.loader import load_all

if __name__ == "__main__":
    data = load_all()
    maintenance = data["maintenance"]
    failures = data["failures"]

    print("=== Maintenance Interval Analysis ===")
    maintenance["datetime"] = pd.to_datetime(maintenance["datetime"])
    maintenance = maintenance.sort_values(["machineID", "datetime"])

    maintenance["prev_maint"] = maintenance.groupby("machineID")["datetime"].shift(1)
    maintenance["interval_days"] = (
        maintenance["datetime"] - maintenance["prev_maint"]
    ).dt.total_seconds() / 86400

    print("\n--- Maintenance Interval Stats (days) ---")
    print(maintenance["interval_days"].describe().round(2))

    print("\n--- Average Interval by Component ---")
    print(maintenance.groupby("comp")["interval_days"].mean().round(2))

    print("\n=== Time Between Failures ===")
    failures["datetime"] = pd.to_datetime(failures["datetime"])
    failures = failures.sort_values(["machineID", "datetime"])
    failures["prev_failure"] = failures.groupby("machineID")["datetime"].shift(1)
    failures["days_since_last_failure"] = (
        failures["datetime"] - failures["prev_failure"]
    ).dt.total_seconds() / 86400

    print("\n--- Time Between Failures Stats (days) ---")
    print(failures["days_since_last_failure"].describe().round(2))

    print("\n--- Average Time Between Failures by Component ---")
    print(failures.groupby("failure")["days_since_last_failure"].mean().round(2))
