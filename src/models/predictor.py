import os
import joblib
import numpy as np
from config import MODELS_DIR


def load_model(model_path=None):
    if model_path is None:
        model_path = os.path.join(MODELS_DIR, "lgbm_pdm.pkl")
    bundle = joblib.load(model_path)
    return bundle["model"], bundle["features"], bundle["label_map"]


def predict(model, X, label_map):
    preds = model.predict(X)
    inv_map = {v: k for k, v in label_map.items()}
    return np.array([inv_map.get(p, "none") for p in preds])


def predict_proba(model, X):
    return model.predict_proba(X)


def predict_single(model, feature_values, feature_cols, label_map):
    import pandas as pd
    row = pd.DataFrame([feature_values], columns=feature_cols)
    pred_enc = model.predict(row.values)[0]
    inv_map = {v: k for k, v in label_map.items()}
    proba = model.predict_proba(row.values)[0]
    return inv_map.get(pred_enc, "none"), dict(zip(inv_map.values(), proba))
