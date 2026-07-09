import pandas as pd


def add_rolling_stats(df, sensor_cols, windows):
    df = df.copy().sort_values(["machineID", "datetime"])
    for window in windows:
        for col in sensor_cols:
            grouped = df.groupby("machineID")[col]
            df[f"{col}_mean_{window}h"] = grouped.transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            df[f"{col}_std_{window}h"] = grouped.transform(
                lambda x: x.rolling(window, min_periods=1).std().fillna(0)
            )
    return df
