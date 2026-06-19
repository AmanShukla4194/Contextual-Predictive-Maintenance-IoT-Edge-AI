import pandas as pd
from config import SENSOR_COLS, ROLLING_WINDOWS
from src.features.rolling_features import add_rolling_stats
from src.features.context_features import time_since_maintenance, cumulative_error_count
from src.features.lag_features import add_lag_features, add_diff_features


def build_full_features(telemetry, maintenance, errors):
    df = add_rolling_stats(telemetry, SENSOR_COLS, ROLLING_WINDOWS)
    df = time_since_maintenance(df, maintenance)
    df = cumulative_error_count(df, errors)
    df = add_lag_features(df, SENSOR_COLS)
    df = add_diff_features(df, SENSOR_COLS)
    df["hours_since_maintenance"] = df["hours_since_maintenance"].fillna(
        df["hours_since_maintenance"].median()
    )
    return df
