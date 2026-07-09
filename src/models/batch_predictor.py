import pandas as pd
import numpy as np
from src.models.predictor import load_model, decode_labels


def predict_batch(df, feature_cols=None, model=None, label_map=None):
    if model is None:
        model, feature_cols, label_map = load_model()

    X = df[feature_cols].values
    preds = model.predict(X)
    probas = model.predict_proba(X)

    results = df[["machineID", "datetime"]].copy() if "datetime" in df.columns else df[["machineID"]].copy()
    results["predicted_label"] = decode_labels(preds, label_map) if label_map else [str(p) for p in preds]
    results["confidence"] = np.round(probas.max(axis=1), 4)
    return results


def predict_latest_per_machine(df, feature_cols=None, model=None, label_map=None):
    latest = (
        df.sort_values("datetime")
        .groupby("machineID")
        .tail(1)
        .reset_index(drop=True)
    )
    results = predict_batch(latest, feature_cols, model, label_map)
    at_risk = results[results["predicted_label"] != "none"].sort_values("confidence", ascending=False)
    return results, at_risk


def summarise_batch_results(results):
    print(f"Total predictions: {len(results)}")
    print(f"Machines flagged: {(results['predicted_label'] != 'none').sum()}")
    print("\nPrediction breakdown:")
    print(results["predicted_label"].value_counts())
    return results["predicted_label"].value_counts().to_dict()
