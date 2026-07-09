import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from src.data.loader import load_all
from src.data.pipeline import build_base_dataset
from src.features.feature_pipeline import build_features
from src.utils.helpers import encode_labels, split_by_time, get_feature_cols
from src.models.predictor import load_model, predict_proba

if __name__ == "__main__":
    data = load_all()
    df = build_base_dataset(data)
    df = build_features(df, data["maintenance"], data["errors"])
    df, label_map = encode_labels(df)

    _, test_df = split_by_time(df)
    model, feature_cols, saved_map = load_model()

    X_test = test_df[feature_cols].values
    y_test = test_df["label_enc"].values
    probas = predict_proba(model, X_test)

    max_conf = probas.max(axis=1)

    print("=== Prediction Confidence Distribution ===")
    bins = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    labels = ["<0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]
    conf_df = pd.cut(max_conf, bins=bins, labels=labels)
    print(conf_df.value_counts().sort_index())

    print(f"\nMean confidence: {max_conf.mean():.4f}")
    print(f"Median confidence: {np.median(max_conf):.4f}")
    print(f"Low-confidence predictions (<0.6): {(max_conf < 0.6).sum()} ({(max_conf < 0.6).mean()*100:.2f}%)")

    print("\n=== Confidence by True Label ===")
    result_df = pd.DataFrame({
        "true_label": [saved_map.get(y, str(y)) for y in y_test],
        "confidence": max_conf
    })
    print(result_df.groupby("true_label")["confidence"].agg(["mean", "min", "max"]).round(4))
