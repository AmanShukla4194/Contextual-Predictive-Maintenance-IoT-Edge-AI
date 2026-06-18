import pandas as pd


def add_lag_features(df, sensor_cols, lags=[1, 3, 6]):
    df = df.copy().sort_values(["machineID", "datetime"])
    for lag in lags:
        for col in sensor_cols:
            df[f"{col}_lag_{lag}h"] = df.groupby("machineID")[col].shift(lag)
    return df.reset_index(drop=True)


def add_diff_features(df, sensor_cols, periods=[1, 3]):
    df = df.copy().sort_values(["machineID", "datetime"])
    for period in periods:
        for col in sensor_cols:
            df[f"{col}_diff_{period}h"] = df.groupby("machineID")[col].diff(period)
    return df.reset_index(drop=True)
