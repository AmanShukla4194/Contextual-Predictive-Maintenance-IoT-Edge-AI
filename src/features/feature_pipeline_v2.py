import pandas as pd
from config import SENSOR_COLS, ROLLING_WINDOWS
from src.features.rolling_features import add_rolling_stats
from src.features.context_features import time_since_maintenance, cumulative_error_count
from src.features.lag_features import add_lag_features, add_diff_features
from src.features.time_features import add_time_features
from src.features.anomaly_score import add_zscore_anomaly, add_anomaly_flag
from src.features.sensor_smoothing import apply_moving_average, apply_exponential_smoothing
from src.features.peak_detection import spike_magnitude


def build_full_features_v2(telemetry, maintenance, errors):
    df = add_rolling_stats(telemetry, SENSOR_COLS, ROLLING_WINDOWS)
    df = time_since_maintenance(df, maintenance)
    df = cumulative_error_count(df, errors)
    df = add_lag_features(df, SENSOR_COLS)
    df = add_diff_features(df, SENSOR_COLS)
    df = add_time_features(df)
    df = add_zscore_anomaly(df, SENSOR_COLS)
    df = add_anomaly_flag(df, SENSOR_COLS)
    df = apply_moving_average(df, SENSOR_COLS, window=3)
    df = apply_exponential_smoothing(df, SENSOR_COLS, alpha=0.3)
    df = spike_magnitude(df, SENSOR_COLS)

    df["hours_since_maintenance"] = df["hours_since_maintenance"].fillna(
        df["hours_since_maintenance"].median()
    )
    numeric_cols = df.select_dtypes(include=["number", "bool"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    return df
