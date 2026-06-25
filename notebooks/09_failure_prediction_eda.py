import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import load_all
from src.data.pipeline import build_base_dataset

if __name__ == "__main__":
    data = load_all()
    df = build_base_dataset(data)

    print("=== Failure Label Distribution ===")
    label_counts = df["failure_label"].value_counts()
    print(label_counts)
    print(f"\nImbalance ratio (none vs all failures): {label_counts['none'] / (len(df) - label_counts['none']):.1f}:1")

    print("\n=== Failure Windows by Component ===")
    failures = df[df["failure_label"] != "none"]
    print(failures["failure_label"].value_counts())

    print("\n=== Average Sensor Readings: Failure vs No Failure ===")
    from config import SENSOR_COLS
    for col in SENSOR_COLS:
        fail_mean = failures[col].mean()
        normal_mean = df[df["failure_label"] == "none"][col].mean()
        print(f"  {col}: failure={fail_mean:.3f}, normal={normal_mean:.3f}, diff={fail_mean - normal_mean:+.3f}")

    print("\n=== Failure Windows by Machine Model ===")
    if "model" in df.columns:
        print(failures.groupby("model")["failure_label"].count())

    print("\n=== Failure Count by Month ===")
    import pandas as pd
    failures["month"] = pd.to_datetime(failures["datetime"]).dt.month
    print(failures.groupby("month")["failure_label"].count())
    