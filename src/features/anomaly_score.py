import pandas as pd
import numpy as np


def add_zscore_anomaly(df, sensor_cols):
    df = df.copy()
    for col in sensor_cols:
        mean = df.groupby("machineID")[col].transform("mean")
        std = df.groupby("machineID")[col].transform("std")
        df[f"{col}_zscore"] = (df[col] - mean) / (std + 1e-9)
    return df


def add_anomaly_flag(df, sensor_cols, threshold=3.0):
    df = df.copy()
    z_cols = [f"{c}_zscore" for c in sensor_cols if f"{c}_zscore" in df.columns]
    if not z_cols:
        df = add_zscore_anomaly(df, sensor_cols)
        z_cols = [f"{c}_zscore" for c in sensor_cols]
    df["anomaly_flag"] = (df[z_cols].abs() > threshold).any(axis=1).astype(int)
    return df


def anomaly_score(df, sensor_cols):
    df = df.copy()
    z_cols = [f"{c}_zscore" for c in sensor_cols if f"{c}_zscore" in df.columns]
    if not z_cols:
        df = add_zscore_anomaly(df, sensor_cols)
        z_cols = [f"{c}_zscore" for c in sensor_cols]
    df["anomaly_score"] = df[z_cols].abs().mean(axis=1)
    return df
