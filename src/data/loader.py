import os
import pandas as pd

_ARCHIVE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "archive",
)


def _read(filename, data_dir, parse_dates=None):
    path = os.path.join(data_dir, filename)
    return pd.read_csv(path, parse_dates=parse_dates)


def load_telemetry(data_dir=_ARCHIVE):
    df = _read("PdM_telemetry.csv", data_dir, parse_dates=["datetime"])
    return df.sort_values(["machineID", "datetime"]).reset_index(drop=True)


def load_errors(data_dir=_ARCHIVE):
    df = _read("PdM_errors.csv", data_dir, parse_dates=["datetime"])
    return df.sort_values(["machineID", "datetime"]).reset_index(drop=True)


def load_failures(data_dir=_ARCHIVE):
    df = _read("PdM_failures.csv", data_dir, parse_dates=["datetime"])
    return df.sort_values(["machineID", "datetime"]).reset_index(drop=True)


def load_maintenance(data_dir=_ARCHIVE):
    df = _read("PdM_maint.csv", data_dir, parse_dates=["datetime"])
    return df.sort_values(["machineID", "datetime"]).reset_index(drop=True)


def load_machines(data_dir=_ARCHIVE):
    return _read("PdM_machines.csv", data_dir)


def load_all(data_dir=_ARCHIVE):
    return {
        "telemetry":   load_telemetry(data_dir),
        "errors":      load_errors(data_dir),
        "failures":    load_failures(data_dir),
        "maintenance": load_maintenance(data_dir),
        "machines":    load_machines(data_dir),
    }
