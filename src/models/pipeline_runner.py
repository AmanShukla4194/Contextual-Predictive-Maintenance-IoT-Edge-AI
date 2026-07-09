import argparse
from src.models.train_pipeline import run_training
from src.models.evaluate_pipeline import run_evaluation
from src.models.model_exporter import export_model_artifacts
from src.models.predictor import load_model
from src.data.loader import load_all
from src.data.pipeline import build_base_dataset
from src.features.feature_pipeline import build_features
from src.utils.helpers import encode_labels


def run_full_pipeline(version=1):
    print("=" * 55)
    print("Step 1: Training")
    print("=" * 55)
    model, feature_cols, label_map = run_training()

    print("\n" + "=" * 55)
    print("Step 2: Evaluation")
    print("=" * 55)
    data = load_all()
    df = build_base_dataset(data)
    df = build_features(df, data["maintenance"], data["errors"])
    df, _ = encode_labels(df)
    results = run_evaluation(df)

    print("\n" + "=" * 55)
    print("Step 3: Export")
    print("=" * 55)
    export_model_artifacts(model, feature_cols, label_map, version=version)

    print("\n" + "=" * 55)
    print(f"Pipeline complete. Macro F1: {results['macro_f1']:.4f}")
    print("=" * 55)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full PdM training pipeline")
    parser.add_argument("--version", type=int, default=1, help="Model version to save")
    args = parser.parse_args()
    run_full_pipeline(version=args.version)
