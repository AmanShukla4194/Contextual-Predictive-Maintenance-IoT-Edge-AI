from src.models.predictor import load_model, predict
from src.models.evaluator import evaluate
from src.utils.helpers import split_by_time, get_feature_cols
from src.data.loader import load_all
from src.data.pipeline import build_base_dataset
from src.features.feature_pipeline import build_features
from src.utils.helpers import encode_labels

MIN_MACRO_F1 = 0.60
MIN_PER_CLASS_F1 = 0.40


def validate_model(min_macro_f1=MIN_MACRO_F1, min_per_class_f1=MIN_PER_CLASS_F1):
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

    passed = True
    print("=== Model Validation Report ===")
    print(f"Macro F1: {results['macro_f1']:.4f} (min: {min_macro_f1})")

    if results["macro_f1"] < min_macro_f1:
        print(f"FAIL: Macro F1 below threshold ({results['macro_f1']:.4f} < {min_macro_f1})")
        passed = False
    else:
        print("PASS: Macro F1 threshold met.")

    for cls, score in results.get("per_class_f1", {}).items():
        status = "PASS" if score >= min_per_class_f1 else "FAIL"
        if score < min_per_class_f1:
            passed = False
        print(f"  [{status}] Class '{cls}': F1 = {score:.4f} (min: {min_per_class_f1})")

    print(f"\nOverall validation: {'PASSED' if passed else 'FAILED'}")
    return passed, results
