import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import load_all
from src.data.eda_utils import failure_distribution, machine_failure_counts
from src.data.preprocessor import label_failures
from config import SENSOR_COLS

if __name__ == "__main__":
    data = load_all()
    failures = data["failures"]
    telemetry = data["telemetry"]

    print("=== Component Failure Counts ===")
    print(failure_distribution(failures))

    print("\n=== Failures by Month ===")
    failures = failures.copy()
    failures["month"] = failures["datetime"].dt.to_period("M")
    print(failures.groupby("month")["failure"].count())

    print("\n=== Top 10 Machines by Failure Count ===")
    print(machine_failure_counts(failures).head(10))

    print("\n=== Average Sensor Readings per Failure Type ===")
    labeled = label_failures(telemetry, data["failures"])
    print(labeled.groupby("label")[SENSOR_COLS].mean().round(2))
