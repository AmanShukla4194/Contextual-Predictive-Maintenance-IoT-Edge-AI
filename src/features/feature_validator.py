import pandas as pd


def check_inf_values(df, feature_cols):
    import numpy as np
    inf_counts = {c: ((df[c] == float("inf")) | (df[c] == float("-inf"))).sum() for c in feature_cols}
    flagged = {k: v for k, v in inf_counts.items() if v > 0}
    if flagged:
        print(f"Inf values found in {len(flagged)} columns: {flagged}")
    else:
        print("No inf values found.")
    return flagged


def check_null_features(df, feature_cols, threshold=0.05):
    null_rates = df[feature_cols].isnull().mean()
    high_null = null_rates[null_rates > threshold]
    if not high_null.empty:
        print(f"High null-rate features (>{threshold*100:.0f}%):")
        print(high_null.round(3))
    else:
        print("All features within null threshold.")
    return high_null


def validate_feature_set(df, feature_cols):
    print("=== Feature Validation ===")
    check_inf_values(df, feature_cols)
    check_null_features(df, feature_cols)
    print(f"Total features: {len(feature_cols)} | Rows: {len(df)}")
