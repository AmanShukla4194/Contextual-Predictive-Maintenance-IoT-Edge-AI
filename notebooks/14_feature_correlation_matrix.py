import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.data.loader import load_all
from src.data.pipeline import build_base_dataset
from src.features.feature_pipeline import build_features
from src.utils.helpers import get_feature_cols

if __name__ == "__main__":
    data = load_all()
    df = build_base_dataset(data)
    df = build_features(df, data["maintenance"], data["errors"])

    feature_cols = get_feature_cols(df)
    print(f"Total features: {len(feature_cols)}")

    print("\n=== Feature Correlation Matrix (top correlated pairs) ===")
    corr = df[feature_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    high_corr = (
        upper.stack()
        .reset_index()
        .rename(columns={"level_0": "feature_1", "level_1": "feature_2", 0: "correlation"})
        .sort_values("correlation", ascending=False)
    )
    print(high_corr.head(20).to_string(index=False))

    print("\n=== Features with Correlation > 0.95 (redundancy candidates) ===")
    redundant = high_corr[high_corr["correlation"] > 0.95]
    print(f"Found {len(redundant)} highly correlated pairs.")
    print(redundant.head(15).to_string(index=False))

    print("\n=== Feature Correlation with Target Label ===")
    if "label_enc" in df.columns:
        target_corr = df[feature_cols + ["label_enc"]].corr()["label_enc"].drop("label_enc")
        print(target_corr.abs().sort_values(ascending=False).head(20).round(4))
