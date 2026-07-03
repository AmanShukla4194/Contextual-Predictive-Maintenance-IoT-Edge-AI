import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.data.loader import load_all

if __name__ == "__main__":
    data = load_all()
    machines = data["machines"]
    failures = data["failures"]
    maintenance = data["maintenance"]

    print("=== Machine Age vs Failure Analysis ===")

    failure_counts = failures.groupby("machineID").size().rename("failure_count").reset_index()
    profile = machines.merge(failure_counts, on="machineID", how="left").fillna(0)
    profile["failure_count"] = profile["failure_count"].astype(int)

    print("\n--- Failure Count by Age Group ---")
    profile["age_group"] = pd.cut(profile["age"], bins=[0, 5, 10, 15, 20], labels=["0-5", "6-10", "11-15", "16-20"])
    print(profile.groupby("age_group")["failure_count"].agg(["mean", "sum", "count"]).round(2))

    print("\n--- Failure Count by Machine Model ---")
    print(profile.groupby("model")["failure_count"].agg(["mean", "sum", "count"]).round(2))

    print("\n--- Correlation: Age vs Failure Count ---")
    corr = profile[["age", "failure_count"]].corr().iloc[0, 1]
    print(f"Pearson correlation: {corr:.4f}")

    print("\n--- Machines with Highest Failure Rates ---")
    print(profile.sort_values("failure_count", ascending=False).head(10)[
        ["machineID", "model", "age", "failure_count"]
    ].to_string(index=False))

    print("\n--- Average Machine Age per Failure Component ---")
    merged = failures.merge(machines, on="machineID", how="left")
    print(merged.groupby("failure")["age"].mean().round(2))