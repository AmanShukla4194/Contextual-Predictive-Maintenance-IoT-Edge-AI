import pandas as pd
import numpy as np
from config import FAILURE_HORIZON, FAILURE_COMPONENTS


def label_failures(telemetry, failures, horizon=FAILURE_HORIZON):
    telemetry = telemetry.copy().sort_values(["machineID", "datetime"]).reset_index(drop=True)
    failures = failures.copy().sort_values(["machineID", "datetime"])
    telemetry["label"] = "none"

    for machine_id, machine_failures in failures.groupby("machineID"):
        machine_idx = telemetry.index[telemetry["machineID"] == machine_id]
        if machine_idx.empty:
            continue

        machine_times = telemetry.loc[machine_idx, "datetime"]
        machine_labels = np.full(len(machine_idx), "none", dtype=object)

        for _, row in machine_failures.iterrows():
            fail_time = row["datetime"]
            window_start = fail_time - pd.Timedelta(hours=horizon)
            in_window = (machine_times >= window_start) & (machine_times < fail_time)
            machine_labels[in_window.to_numpy()] = row["failure"]

        telemetry.loc[machine_idx, "label"] = machine_labels

    return telemetry


def merge_machine_info(df, machines):
    model_dummies = pd.get_dummies(machines["model"], prefix="model")
    machines_encoded = pd.concat(
        [machines[["machineID", "age"]], model_dummies], axis=1
    )
    return df.merge(machines_encoded, on="machineID", how="left")
