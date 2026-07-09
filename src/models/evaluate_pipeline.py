import os
from config import REPORTS_DIR
from src.models.predictor import load_model, predict
from src.models.evaluator import evaluate
from src.utils.helpers import split_by_time


def run_evaluation(df):
    model, feature_cols, label_map = load_model()

    _, test_df = split_by_time(df)
    X_test = test_df[feature_cols].values
    y_test = test_df["label_enc"].values

    predictions = model.predict(X_test)
    results = evaluate(y_test, predictions, label_map)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(REPORTS_DIR, "evaluation_results.txt")
    with open(report_path, "w") as f:
        f.write(f"Macro F1: {results['macro_f1']:.4f}\n\n")
        f.write(results["report"])

    print(f"Evaluation report saved to {report_path}")
    return results
