import pandas as pd


def apply_moving_average(df, sensor_cols, window=3):
    df = df.copy().sort_values(["machineID", "datetime"])
    for col in sensor_cols:
        df[f"{col}_sma_{window}h"] = (
            df.groupby("machineID")[col]
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
    return df


def apply_exponential_smoothing(df, sensor_cols, alpha=0.3):
    df = df.copy().sort_values(["machineID", "datetime"])
    for col in sensor_cols:
        df[f"{col}_ema"] = (
            df.groupby("machineID")[col]
            .transform(lambda x: x.ewm(alpha=alpha, adjust=False).mean())
        )
    return df


def apply_savgol_smoothing(df, sensor_cols, window=7, polyorder=2):
    from scipy.signal import savgol_filter
    df = df.copy().sort_values(["machineID", "datetime"])
    for col in sensor_cols:
        df[f"{col}_savgol"] = (
            df.groupby("machineID")[col]
            .transform(lambda x: savgol_filter(x, window_length=min(window, len(x)), polyorder=polyorder)
                       if len(x) >= window else x)
        )
    return df
