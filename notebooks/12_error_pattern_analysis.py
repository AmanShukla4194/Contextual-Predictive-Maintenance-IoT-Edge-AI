import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.data.loader import load_all

if __name__ == "__main__":
    data = load_all()
    errors = data["errors"]
    failures = data["failures"]

    errors["datetime"] = pd.to_datetime(errors["datetime"])
    failures["datetime"] = pd.to_datetime(failures["datetime"])

    print("=== Error Pattern Analysis ===")

    print("\n--- Overall Error Frequency ---")
    print(errors["errorID"].value_counts())

    print("\n--- Errors Per Machine (top 10) ---")
    print(errors.groupby("machineID").size().sort_values(ascending=False).head(10))

    print("\n--- Errors in 24h Before a Failure ---")
    pre_failure_errors = []
    for _, failure_row in failures.iterrows():
        mid = failure_row["machineID"]
        ft = failure_row["datetime"]
        window = errors[
            (errors["machineID"] == mid) &
            (errors["datetime"] >= ft - pd.Timedelta(hours=24)) &
            (errors["datetime"] < ft)
        ]
        for _, err_row in window.iterrows():
            pre_failure_errors.append({
                "machineID": mid,
                "failure": failure_row["failure"],
                "errorID": err_row["errorID"],
                "hours_before": round((ft - err_row["datetime"]).total_seconds() / 3600, 2)
            })

    pre_df = pd.DataFrame(pre_failure_errors)
    if not pre_df.empty:
        print(pre_df.groupby(["failure", "errorID"]).size().sort_values(ascending=False).head(20))
        print(f"\nAverage errors in 24h pre-failure: {pre_df.groupby('machineID').size().mean():.2f}")
    else:
        print("No errors found in 24h pre-failure windows.")
