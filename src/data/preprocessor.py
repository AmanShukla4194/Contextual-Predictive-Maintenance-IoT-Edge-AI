import pandas as pd
import numpy as np
from config import FAILURE_HORIZON, FAILURE_COMPONENTS


def label_failures(telemetry, failures, horizon=FAILURE_HORIZON):
    telemetry = telemetry.copy()
    telemetry["label"] = "none"

    for _, row in failures.iterrows():
        machine = row["machineID"]
        fail_time = row["datetime"]
        component = row["failure"]
        window_start = fail_time - pd.Timedelta(hours=horizon)

        mask = (
            (telemetry["machineID"] == machine)
            & (telemetry["datetime"] >= window_start)
            & (telemetry["datetime"] < fail_time)
        )
        telemetry.loc[mask, "label"] = component

    return telemetry


def merge_machine_info(df, machines):
    model_dummies = pd.get_dummies(machines["model"], prefix="model")
    machines_encoded = pd.concat(
        [machines[["machineID", "age"]], model_dummies], axis=1
    )
    return df.merge(machines_encoded, on="machineID", how="left")
