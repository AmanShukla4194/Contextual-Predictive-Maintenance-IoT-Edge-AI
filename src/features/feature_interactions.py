import pandas as pd


def add_sensor_ratios(df, sensor_cols):
    df = df.copy()
    for i in range(len(sensor_cols)):
        for j in range(i + 1, len(sensor_cols)):
            a, b = sensor_cols[i], sensor_cols[j]
            df[f"{a}_div_{b}"] = df[a] / (df[b] + 1e-9)
    return df


def add_sensor_products(df, sensor_pairs):
    df = df.copy()
    for a, b in sensor_pairs:
        df[f"{a}_x_{b}"] = df[a] * df[b]
    return df


def add_deviation_from_mean(df, sensor_cols):
    df = df.copy()
    for col in sensor_cols:
        group_mean = df.groupby("machineID")[col].transform("mean")
        df[f"{col}_dev"] = df[col] - group_mean
    return df
