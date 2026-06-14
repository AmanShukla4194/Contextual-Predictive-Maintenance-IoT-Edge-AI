import pandas as pd


def merge_errors_to_telemetry(telemetry, errors):
    errors = errors.copy()
    errors["datetime"] = errors["datetime"].dt.floor("h")
    error_dummies = pd.get_dummies(errors["errorID"], prefix="error")
    errors = pd.concat([errors[["machineID", "datetime"]], error_dummies], axis=1)
    error_counts = errors.groupby(["machineID", "datetime"]).sum().reset_index()
    return telemetry.merge(error_counts, on=["machineID", "datetime"], how="left").fillna(0)


def merge_maintenance_to_telemetry(telemetry, maintenance):
    maintenance = maintenance.copy()
    maintenance["datetime"] = maintenance["datetime"].dt.floor("h")
    maint_dummies = pd.get_dummies(maintenance["comp"], prefix="maint")
    maintenance = pd.concat([maintenance[["machineID", "datetime"]], maint_dummies], axis=1)
    maint_counts = maintenance.groupby(["machineID", "datetime"]).sum().reset_index()
    return telemetry.merge(maint_counts, on=["machineID", "datetime"], how="left").fillna(0)