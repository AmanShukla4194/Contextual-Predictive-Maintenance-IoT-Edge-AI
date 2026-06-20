import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import load_all
from config import SENSOR_COLS

if __name__ == "__main__":
    data = load_all()
    telemetry = data["telemetry"]
    machines = data["machines"]
    failures = data["failures"]
    maintenance = data["maintenance"]

    print("=== Machine Fleet Profile ===")
    profile = machines.copy()
    failure_counts = failures.groupby("machineID").size().rename("total_failures")
    maint_counts = maintenance.groupby("machineID").size().rename("total_maintenance")
    profile = profile.merge(failure_counts, on="machineID", how="left")
    profile = profile.merge(maint_counts, on="machineID", how="left")
    profile = profile.fillna(0)
    print(profile.head(20))

    print("\n=== Average Sensor Readings per Machine ===")
    avg_sensors = telemetry.groupby("machineID")[SENSOR_COLS].mean().round(2)
    print(avg_sensors.head(10))

    print("\n=== Machines with Zero Failures ===")
    no_failures = set(machines["machineID"]) - set(failures["machineID"])
    print(f"{len(no_failures)} machines recorded no failures: {sorted(no_failures)}")