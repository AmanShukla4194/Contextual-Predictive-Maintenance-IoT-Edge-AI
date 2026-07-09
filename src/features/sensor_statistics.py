import pandas as pd
import numpy as np


def per_machine_sensor_stats(df, sensor_cols):
    stats = df.groupby("machineID")[sensor_cols].agg(["mean", "std", "min", "max"])
    stats.columns = ["_".join(c) for c in stats.columns]
    return stats.reset_index()


def sensor_trend_direction(df, sensor_cols, window=24):
    df = df.copy().sort_values(["machineID", "datetime"])
    for col in sensor_cols:
        rolling_mean = df.groupby("machineID")[col].transform(
            lambda x: x.rolling(window, min_periods=2).mean()
        )
        shifted = df.groupby("machineID")[col].transform(
            lambda x: x.rolling(window, min_periods=2).mean().shift(window)
        )
        df[f"{col}_trend"] = np.sign(rolling_mean - shifted)
    return df


def coefficient_of_variation(df, sensor_cols):
    cv = {}
    for col in sensor_cols:
        mean = df[col].mean()
        std = df[col].std()
        cv[col] = round(std / (abs(mean) + 1e-9), 4)
    return cv
