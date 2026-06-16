import pandas as pd
import numpy as np


def drop_duplicates(df, subset=None):
    before = len(df)
    df = df.drop_duplicates(subset=subset).reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        print(f"Dropped {dropped} duplicate rows")
    return df


def fill_missing_sensors(df, sensor_cols):
    df = df.copy()
    df[sensor_cols] = df.groupby("machineID")[sensor_cols].transform(
        lambda x: x.ffill().bfill()
    )
    return df


def remove_outliers_iqr(df, cols, factor=3.0):
    df = df.copy()
    for col in cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        df[col] = df[col].clip(lower, upper)
    return df