import pandas as pd
import numpy as np


def time_since_maintenance(telemetry, maintenance):
    telemetry = telemetry.copy().sort_values(["machineID", "datetime"])
    maintenance = maintenance.sort_values(["machineID", "datetime"])

    records = []
    for machine_id, group in telemetry.groupby("machineID"):
        maint = maintenance[maintenance["machineID"] == machine_id]["datetime"].values
        tel_times = group["datetime"].values
        diffs = []
        for t in tel_times:
            past = maint[maint <= t]
            diffs.append(
                float((t - past[-1]) / np.timedelta64(1, "h")) if len(past) > 0 else np.nan
            )
        group = group.copy()
        group["hours_since_maintenance"] = diffs
        records.append(group)

    return pd.concat(records).sort_values(["machineID", "datetime"]).reset_index(drop=True)


def cumulative_error_count(telemetry, errors):
    telemetry = telemetry.copy().sort_values(["machineID", "datetime"])
    errors = errors.sort_values(["machineID", "datetime"])

    result_frames = []
    for machine_id, group in telemetry.groupby("machineID"):
        errs = errors[errors["machineID"] == machine_id]["datetime"].values
        counts = [int((errs <= t).sum()) for t in group["datetime"].values]
        group = group.copy()
        group["cumulative_errors"] = counts
        result_frames.append(group)

    return pd.concat(result_frames).sort_values(["machineID", "datetime"]).reset_index(drop=True)
