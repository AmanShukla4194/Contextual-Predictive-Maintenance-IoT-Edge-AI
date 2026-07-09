import os
import json
import joblib
from config import MODELS_DIR


def export_model_artifacts(model, feature_cols, label_map, version=1):
    os.makedirs(MODELS_DIR, exist_ok=True)

    model_path = os.path.join(MODELS_DIR, "lgbm_pdm.pkl")
    joblib.dump({"model": model, "feature_cols": feature_cols, "label_map": label_map}, model_path)
    print(f"Model saved: {model_path}")

    features_path = os.path.join(MODELS_DIR, "feature_cols.json")
    with open(features_path, "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"Feature list saved: {features_path}")

    label_path = os.path.join(MODELS_DIR, "label_map.json")
    with open(label_path, "w") as f:
        json.dump({str(k): v for k, v in label_map.items()}, f, indent=2)
    print(f"Label map saved: {label_path}")

    meta = {
        "version": version,
        "num_features": len(feature_cols),
        "num_classes": len(label_map),
        "classes": list(label_map.values()),
    }
    meta_path = os.path.join(MODELS_DIR, "export_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Export metadata saved: {meta_path}")
    return model_path


def load_exported_artifacts():
    model_path = os.path.join(MODELS_DIR, "lgbm_pdm.pkl")
    features_path = os.path.join(MODELS_DIR, "feature_cols.json")
    label_path = os.path.join(MODELS_DIR, "label_map.json")

    data = joblib.load(model_path)
    with open(features_path) as f:
        feature_cols = json.load(f)
    with open(label_path) as f:
        label_map = {int(k): v for k, v in json.load(f).items()}

    return data["model"], feature_cols, label_map
