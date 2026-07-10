import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.data.loader import load_all
from src.data.pipeline import build_base_dataset
from src.features.feature_pipeline import build_features
from src.utils.helpers import encode_labels, split_by_time, get_feature_cols
from config import SENSOR_COLS

if __name__ == "__main__":
    data = load_all()

    print("=" * 55)
    print("PROJECT SUMMARY — Contextual Predictive Maintenance")
    print("=" * 55)

    print("\n--- Dataset Overview ---")
    print(f"  Machines       : {data['machines']['machineID'].nunique()}")
    print(f"  Telemetry rows : {len(data['telemetry']):,}")
    print(f"  Failures       : {len(data['failures'])}")
    print(f"  Errors         : {len(data['errors'])}")
    print(f"  Maintenance    : {len(data['maintenance'])}")

    df = build_base_dataset(data)
    df = build_features(df, data["maintenance"], data["errors"])
    df, label_map = encode_labels(df)
    feature_cols = get_feature_cols(df)

    print("\n--- Feature Engineering ---")
    print(f"  Total features : {len(feature_cols)}")
    print(f"  Rolling feats  : {sum('_mean_' in c or '_std_' in c for c in feature_cols)}")
    print(f"  Lag feats      : {sum('_lag_' in c for c in feature_cols)}")
    print(f"  Diff feats     : {sum('_diff_' in c for c in feature_cols)}")

    print("\n--- Label Distribution ---")
    label_dist = df["failure_label"].value_counts()
    for label, count in label_dist.items():
        print(f"  {label:<10}: {count:>7,} ({count/len(df)*100:.2f}%)")

    train_df, test_df = split_by_time(df)
    print(f"\n--- Train/Test Split ---")
    print(f"  Train rows : {len(train_df):,}")
    print(f"  Test rows  : {len(test_df):,}")
    print(f"  Date range : {df['datetime'].min().date()} to {df['datetime'].max().date()}")

    print("\n--- Sensor Summary (global) ---")
    print(df[SENSOR_COLS].describe().T[["mean", "std", "min", "max"]].round(3))

    print("\n--- Key Findings ---")
    print("  1. Failures are rare events (~3-5% of records) — imbalance handled via SMOTE")
    print("  2. Rolling 24h and 72h features carry the highest predictive signal")
    print("  3. Time since maintenance is strongly correlated with comp3/comp4 failures")
    print("  4. Error frequency in 24h pre-failure window is a reliable early warning signal")
    print("  5. LightGBM with class_weight=balanced achieves best macro F1 on all 5 classes")