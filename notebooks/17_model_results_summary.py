import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import load_all
from src.data.pipeline import build_base_dataset
from src.features.feature_pipeline import build_features
from src.utils.helpers import encode_labels, split_by_time, get_feature_cols
from src.models.predictor import load_model, predict
from src.models.evaluator import evaluate
from src.models.model_registry import list_versions

if __name__ == "__main__":
    print("=== Model Results Summary ===")

    print("\n--- Saved Model Versions ---")
    versions = list_versions()
    if versions:
        for v in versions:
            print(f"  v{v['version']}: Macro F1 = {v['macro_f1']:.4f} | Features = {v['feature_count']}")
    else:
        print("  No versioned models found.")

    print("\n--- Loading Best Model & Evaluating on Test Set ---")
    data = load_all()
    df = build_base_dataset(data)
    df = build_features(df, data["maintenance"], data["errors"])
    df, label_map = encode_labels(df)
    _, test_df = split_by_time(df)

    model, feature_cols, saved_map = load_model()
    X_test = test_df[feature_cols].values
    y_test = test_df["label_enc"].values
    preds = predict(model, X_test)
    results = evaluate(y_test, preds, saved_map)

    print(f"\nTest Macro F1:  {results['macro_f1']:.4f}")
    print(f"Total features: {len(feature_cols)}")
    print(f"Test samples:   {len(y_test)}")

    print("\n--- Full Classification Report ---")
    print(results["report"])

    print("\n--- Top 15 Features by Importance ---")
    import numpy as np
    imp = model.feature_importances_
    top_idx = np.argsort(imp)[::-1][:15]
    for rank, i in enumerate(top_idx, 1):
        print(f"  {rank:>2}. {feature_cols[i]:<45} {imp[i]:.1f}")
