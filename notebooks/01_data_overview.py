import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import load_all
from src.data.validator import summarise
from src.data.eda_utils import (
    failure_distribution,
    machine_failure_counts,
    telemetry_stats,
    error_frequency,
)
from config import SENSOR_COLS

if __name__ == "__main__":
    print("Loading datasets...")
    data = load_all()

    print("\n=== Dataset Summary ===")
    summarise(data)

    print("\n=== Failure Distribution ===")
    print(failure_distribution(data["failures"]))

    print("\n=== Top 10 Machines by Failure Count ===")
    print(machine_failure_counts(data["failures"]).head(10))

    print("\n=== Telemetry Sensor Statistics ===")
    print(telemetry_stats(data["telemetry"], SENSOR_COLS))

    print("\n=== Error Frequency ===")
    print(error_frequency(data["errors"]))
