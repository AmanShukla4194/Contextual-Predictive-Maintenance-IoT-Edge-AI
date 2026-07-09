import os
import json
import numpy as np
from config import REPORTS_DIR
from src.data.loader import load_all
from src.data.pipeline import build_base_dataset
from src.features.feature_pipeline import build_features
from src.utils.helpers import encode_labels, split_by_time, get_feature_cols
from src.models.predictor import load_model, predict, predict_proba
from src.models.evaluator import evaluate


def generate_final_report():
    data = load_all()
    df = build_base_dataset(data)
    df = build_features(df, data["maintenance"], data["errors"])
    df, label_map = encode_labels(df)

    train_df, test_df = split_by_time(df)
    model, feature_cols, saved_map = load_model()

    X_test = test_df[feature_cols].values
    y_test = test_df["label_enc"].values
    preds = predict(model, X_test)
    probas = predict_proba(model, X_test)
    results = evaluate(y_test, preds, saved_map)

    imp = model.feature_importances_
    top_idx = np.argsort(imp)[::-1][:20]
    top_features = {feature_cols[i]: round(float(imp[i]), 4) for i in top_idx}

    report = {
        "macro_f1": round(results["macro_f1"], 4),
        "total_features": len(feature_cols),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "mean_confidence": round(float(probas.max(axis=1).mean()), 4),
        "per_class_f1": results.get("per_class_f1", {}),
        "top_20_features": top_features,
        "classification_report": results["report"],
    }

    os.makedirs(REPORTS_DIR, exist_ok=True)
    out_path = os.path.join(REPORTS_DIR, "final_model_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Final model report saved to {out_path}")
    print(f"Macro F1: {report['macro_f1']:.4f}")
    print(f"Total features: {report['total_features']}")
    print(f"Mean confidence: {report['mean_confidence']:.4f}")
    return report


if __name__ == "__main__":
    generate_final_report()
