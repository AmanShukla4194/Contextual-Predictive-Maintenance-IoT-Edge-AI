import pandas as pd
import numpy as np


def detect_noisy_features(df, feature_cols, cv_threshold=0.01):
    stats = df[feature_cols].describe().T
    stats["cv"] = stats["std"] / (stats["mean"].abs() + 1e-9)
    return stats[stats["cv"] < cv_threshold].index.tolist()


def drop_noisy_features(df, feature_cols, cv_threshold=0.01):
    noisy = detect_noisy_features(df, feature_cols, cv_threshold)
    keep = [c for c in feature_cols if c not in noisy]
    if noisy:
        print(f"Dropped {len(noisy)} noisy features: {noisy}")
    return df[[c for c in df.columns if c not in noisy]], keep


def clip_sensor_extremes(df, sensor_cols, lower_pct=0.01, upper_pct=0.99):
    df = df.copy()
    for col in sensor_cols:
        lo = df[col].quantile(lower_pct)
        hi = df[col].quantile(upper_pct)
        df[col] = df[col].clip(lo, hi)
    return df
