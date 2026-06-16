from config import SENSOR_COLS, ROLLING_WINDOWS
from src.features.rolling_features import add_rolling_stats
from src.features.context_features import time_since_maintenance, cumulative_error_count


def build_features(telemetry, maintenance, errors):
    df = add_rolling_stats(telemetry, SENSOR_COLS, ROLLING_WINDOWS)
    df = time_since_maintenance(df, maintenance)
    df = cumulative_error_count(df, errors)
    df["hours_since_maintenance"] = df["hours_since_maintenance"].fillna(
        df["hours_since_maintenance"].median()
    )
    return df
