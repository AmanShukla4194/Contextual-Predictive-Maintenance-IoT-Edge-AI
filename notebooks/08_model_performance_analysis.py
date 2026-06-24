import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import load_all
from src.data.pipeline import build_base_dataset
from src.features.feature_pipeline import build_features
from src.utils.helpers import encode_labels, split_by_time, get_feature_cols
from src.models.predictor import load_model, predict
from src.models.evaluator import evaluate

if __name__ == "__main__":
    print("=== Model Performance Analysis ===")

    data = load_all()
    df = build_base_dataset(data)
    df = build_features(df, data["maintenance"], data["errors"])
    df, label_map = encode_labels(df)

    _, test_df = split_by_time(df)
    feature_cols = get_feature_cols(test_df)

    model, saved_cols, saved_map = load_model()
    X_test = test_df[saved_cols].values
    y_test = test_df["label_enc"].values

    preds = predict(model, X_test)
    results = evaluate(y_test, preds, saved_map)

    print(f"\nMacro F1 on test set: {results['macro_f1']:.4f}")
    print("\nClassification Report:")
    print(results["report"])

    print("\nPer-class breakdown:")
    for cls, score in results.get("per_class_f1", {}).items():
        print(f"  {cls}: {score:.4f}")