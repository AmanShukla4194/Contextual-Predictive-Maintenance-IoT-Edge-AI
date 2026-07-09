import numpy as np

from config import SENSOR_COLS, ROLLING_WINDOWS
from src.features.rolling_features import add_rolling_stats
from src.features.context_features import time_since_maintenance, cumulative_error_count
from src.features.lag_features import add_lag_features, add_diff_features
from src.features.time_features import add_time_features
from src.features.feature_interactions import add_sensor_ratios, add_deviation_from_mean


def build_features(telemetry, maintenance, errors):
    df = add_rolling_stats(telemetry, SENSOR_COLS, ROLLING_WINDOWS)
    df = time_since_maintenance(df, maintenance)
    df = cumulative_error_count(df, errors)
    df = add_lag_features(df, SENSOR_COLS)
    df = add_diff_features(df, SENSOR_COLS)
    df = add_time_features(df)
    df = add_sensor_ratios(df, SENSOR_COLS)
    df = add_deviation_from_mean(df, SENSOR_COLS)

    df["hours_since_maintenance"] = df["hours_since_maintenance"].fillna(
        df["hours_since_maintenance"].median()
    )
    df = df.replace([np.inf, -np.inf], np.nan)
    numeric_cols = df.select_dtypes(include=["number", "bool"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    return df
