import pandas as pd


def add_rolling_min_max(df, sensor_cols, windows):
    df = df.copy()
    df = df.sort_values(["machineID", "datetime"])
    for col in sensor_cols:
        for w in windows:
            grp = df.groupby("machineID")[col]
            df[f"{col}_min_{w}h"] = grp.transform(lambda x: x.rolling(w, min_periods=1).min())
            df[f"{col}_max_{w}h"] = grp.transform(lambda x: x.rolling(w, min_periods=1).max())
            df[f"{col}_range_{w}h"] = df[f"{col}_max_{w}h"] - df[f"{col}_min_{w}h"]
    return df


def add_rolling_median(df, sensor_cols, windows):
    df = df.copy()
    df = df.sort_values(["machineID", "datetime"])
    for col in sensor_cols:
        for w in windows:
            grp = df.groupby("machineID")[col]
            df[f"{col}_median_{w}h"] = grp.transform(lambda x: x.rolling(w, min_periods=1).median())
    return df


def add_ewm_features(df, sensor_cols, spans=None):
    if spans is None:
        spans = [6, 24]
    df = df.copy()
    df = df.sort_values(["machineID", "datetime"])
    for col in sensor_cols:
        for span in spans:
            df[f"{col}_ewm_{span}h"] = (
                df.groupby("machineID")[col]
                .transform(lambda x: x.ewm(span=span, adjust=False).mean())
            )
    return df
