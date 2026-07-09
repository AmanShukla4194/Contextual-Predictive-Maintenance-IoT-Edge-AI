import numpy as np
import pandas as pd


def remove_low_variance(df, feature_cols, threshold=0.01):
    variances = df[feature_cols].var()
    keep = variances[variances > threshold].index.tolist()
    dropped = [c for c in feature_cols if c not in keep]
    if dropped:
        print(f"Dropped {len(dropped)} low-variance features: {dropped}")
    return keep


def correlation_filter(df, feature_cols, threshold=0.95):
    corr_matrix = df[feature_cols].corr().abs()
    upper_tri = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    upper = corr_matrix.where(upper_tri)
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    keep = [c for c in feature_cols if c not in to_drop]
    if to_drop:
        print(f"Removed {len(to_drop)} highly correlated features: {to_drop}")
    return keep
