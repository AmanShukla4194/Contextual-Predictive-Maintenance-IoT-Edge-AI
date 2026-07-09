import pandas as pd
import numpy as np


def detect_peaks(df, sensor_cols, order=3):
    from scipy.signal import argrelextrema
    df = df.copy().sort_values(["machineID", "datetime"])
    for col in sensor_cols:
        peak_flags = np.zeros(len(df), dtype=int)
        for _, grp in df.groupby("machineID"):
            idx = grp.index.tolist()
            vals = grp[col].values
            if len(vals) > 2 * order:
                local_max = argrelextrema(vals, np.greater, order=order)[0]
                for i in local_max:
                    peak_flags[df.index.get_loc(idx[i])] = 1
        df[f"{col}_peak"] = peak_flags
    return df


def spike_magnitude(df, sensor_cols, window=6):
    df = df.copy().sort_values(["machineID", "datetime"])
    for col in sensor_cols:
        rolling_mean = df.groupby("machineID")[col].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[f"{col}_spike"] = (df[col] - rolling_mean).abs()
    return df


def count_spikes(df, sensor_cols, threshold_std=2.0):
    df = df.copy()
    for col in sensor_cols:
        spike_col = f"{col}_spike"
        if spike_col not in df.columns:
            df = spike_magnitude(df, [col])
        std_val = df[spike_col].std()
        df[f"{col}_spike_flag"] = (df[spike_col] > threshold_std * std_val).astype(int)
    return df
