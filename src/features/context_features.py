import pandas as pd
import numpy as np


def time_since_maintenance(telemetry, maintenance):
    telemetry = telemetry.copy().sort_values(["machineID", "datetime"])
    maintenance = maintenance.sort_values(["machineID", "datetime"])

    records = []
    for machine_id, group in telemetry.groupby("machineID"):
        maint = maintenance[maintenance["machineID"] == machine_id]["datetime"].to_numpy(dtype="datetime64[ns]")
        tel_times = group["datetime"].to_numpy(dtype="datetime64[ns]")
        diffs = np.full(len(group), np.nan, dtype=float)
        if len(maint) > 0:
            positions = np.searchsorted(maint, tel_times, side="right") - 1
            valid = positions >= 0
            diffs[valid] = (
                tel_times[valid] - maint[positions[valid]]
            ) / np.timedelta64(1, "h")
        group = group.copy()
        group["hours_since_maintenance"] = diffs
        records.append(group)

    return pd.concat(records).sort_values(["machineID", "datetime"]).reset_index(drop=True)


def cumulative_error_count(telemetry, errors):
    telemetry = telemetry.copy().sort_values(["machineID", "datetime"])
    errors = errors.sort_values(["machineID", "datetime"])

    result_frames = []
    for machine_id, group in telemetry.groupby("machineID"):
        errs = errors[errors["machineID"] == machine_id]["datetime"].to_numpy(dtype="datetime64[ns]")
        tel_times = group["datetime"].to_numpy(dtype="datetime64[ns]")
        counts = np.searchsorted(errs, tel_times, side="right").astype(int)
        group = group.copy()
        group["cumulative_error_count"] = counts
        result_frames.append(group)

    return pd.concat(result_frames).sort_values(["machineID", "datetime"]).reset_index(drop=True)
