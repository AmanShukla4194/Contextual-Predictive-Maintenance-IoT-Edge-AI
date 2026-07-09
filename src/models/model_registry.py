import os
import json
import joblib
from config import MODELS_DIR


def save_model_version(model, feature_cols, label_map, version, score):
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"lgbm_v{version}.pkl")
    meta_path = os.path.join(MODELS_DIR, f"lgbm_v{version}_meta.json")

    joblib.dump({"model": model, "feature_cols": feature_cols, "label_map": label_map}, model_path)

    meta = {"version": version, "macro_f1": round(score, 4), "feature_count": len(feature_cols)}
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved model version {version} | Macro F1: {score:.4f} -> {model_path}")


def load_model_version(version):
    model_path = os.path.join(MODELS_DIR, f"lgbm_v{version}.pkl")
    data = joblib.load(model_path)
    return data["model"], data["feature_cols"], data["label_map"]


def list_versions():
    if not os.path.exists(MODELS_DIR):
        return []
    entries = []
    for f in os.listdir(MODELS_DIR):
        if f.endswith("_meta.json"):
            with open(os.path.join(MODELS_DIR, f)) as fp:
                entries.append(json.load(fp))
    return sorted(entries, key=lambda x: x["version"])
