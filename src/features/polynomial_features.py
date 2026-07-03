import pandas as pd


def add_squared_features(df, sensor_cols):
    df = df.copy()
    for col in sensor_cols:
        df[f"{col}_sq"] = df[col] ** 2
    return df


def add_cubed_features(df, sensor_cols):
    df = df.copy()
    for col in sensor_cols:
        df[f"{col}_cb"] = df[col] ** 3
    return df


def add_sqrt_features(df, sensor_cols):
    import numpy as np
    df = df.copy()
    for col in sensor_cols:
        df[f"{col}_sqrt"] = np.sqrt(df[col].clip(0))
    return df


def add_log_features(df, sensor_cols):
    import numpy as np
    df = df.copy()
    for col in sensor_cols:
        df[f"{col}_log"] = np.log1p(df[col].clip(0))
    return df
