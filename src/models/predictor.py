import os
import joblib
import numpy as np
from config import MODELS_DIR


def inverse_label_map(label_map):
    if not label_map:
        return {}

    first_key = next(iter(label_map.keys()))
    first_value = label_map[first_key]

    if isinstance(first_key, str) and not str(first_key).isdigit() and isinstance(first_value, (int, np.integer)):
        return {int(v): str(k) for k, v in label_map.items()}

    inverse = {}
    for key, value in label_map.items():
        try:
            inverse[int(key)] = str(value)
        except (TypeError, ValueError):
            try:
                inverse[int(value)] = str(key)
            except (TypeError, ValueError):
                inverse[key] = str(value)
    return inverse


def decode_labels(predictions, label_map):
    inv_map = inverse_label_map(label_map)
    return np.array([inv_map.get(int(p), str(p)) if isinstance(p, (int, np.integer)) else inv_map.get(p, str(p)) for p in predictions])


def probability_by_label(model, proba, label_map):
    inv_map = inverse_label_map(label_map)
    classes = getattr(model, "classes_", np.arange(len(proba)))
    return {
        inv_map.get(int(cls), str(cls)): float(proba[i])
        for i, cls in enumerate(classes)
    }


def load_model(model_path=None):
    if model_path is None:
        model_path = os.path.join(MODELS_DIR, "lgbm_pdm.pkl")
    bundle = joblib.load(model_path)
    feature_cols = bundle.get("features") or bundle.get("feature_cols")
    if feature_cols is None:
        raise KeyError("Model artifact is missing a 'features' or 'feature_cols' list.")
    return bundle["model"], feature_cols, bundle["label_map"]


def predict(model, X, label_map):
    preds = model.predict(X)
    return decode_labels(preds, label_map)


def predict_proba(model, X):
    return model.predict_proba(X)


def predict_single(model, feature_values, feature_cols, label_map):
    import pandas as pd
    row = pd.DataFrame([feature_values], columns=feature_cols)
    pred_enc = model.predict(row.values)[0]
    proba = model.predict_proba(row.values)[0]
    return decode_labels([pred_enc], label_map)[0], probability_by_label(model, proba, label_map)


def prepare_feature_frame(telemetry_df=None, feature_cols=None, data=None):
    from src.data.loader import load_all
    from src.data.merger import merge_errors_to_telemetry, merge_maintenance_to_telemetry
    from src.data.preprocessor import merge_machine_info
    from src.features.feature_pipeline import build_features

    if data is None:
        data = load_all()
    if telemetry_df is None:
        telemetry_df = data["telemetry"]

    df = telemetry_df.copy()
    if "error_error1" not in df.columns:
        df = merge_errors_to_telemetry(df, data["errors"])
    if "maint_comp1" not in df.columns:
        df = merge_maintenance_to_telemetry(df, data["maintenance"])
    if "age" not in df.columns:
        df = merge_machine_info(df, data["machines"])

    df = build_features(df, data["maintenance"], data["errors"])

    if feature_cols:
        missing = [c for c in feature_cols if c not in df.columns]
        for col in missing:
            df[col] = 0.0
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    return df
