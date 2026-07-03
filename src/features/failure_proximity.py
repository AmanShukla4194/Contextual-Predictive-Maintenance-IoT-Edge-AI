import pandas as pd
import numpy as np


def time_since_last_failure(telemetry, failures):
    telemetry = telemetry.copy().sort_values(["machineID", "datetime"])
    failures = failures.copy()
    failures["datetime"] = pd.to_datetime(failures["datetime"])
    telemetry["datetime"] = pd.to_datetime(telemetry["datetime"])

    telemetry["hours_since_last_failure"] = np.nan

    for mid, grp in telemetry.groupby("machineID"):
        machine_failures = failures[failures["machineID"] == mid]["datetime"].sort_values().values
        if len(machine_failures) == 0:
            continue
        for idx, row_time in grp["datetime"].items():
            past = machine_failures[machine_failures <= np.datetime64(row_time)]
            if len(past) > 0:
                last = pd.Timestamp(past[-1])
                telemetry.at[idx, "hours_since_last_failure"] = (
                    row_time - last
                ).total_seconds() / 3600

    return telemetry


def failure_count_rolling(telemetry, failures, window_hours=168):
    telemetry = telemetry.copy().sort_values(["machineID", "datetime"])
    telemetry["datetime"] = pd.to_datetime(telemetry["datetime"])
    failures["datetime"] = pd.to_datetime(failures["datetime"])
    telemetry["recent_failure_count"] = 0

    for mid, grp in telemetry.groupby("machineID"):
        machine_failures = failures[failures["machineID"] == mid]["datetime"].values
        if len(machine_failures) == 0:
            continue
        for idx, row_time in grp["datetime"].items():
            cutoff = row_time - pd.Timedelta(hours=window_hours)
            count = ((machine_failures >= np.datetime64(cutoff)) & (machine_failures < np.datetime64(row_time))).sum()
            telemetry.at[idx, "recent_failure_count"] = int(count)

    return telemetry
