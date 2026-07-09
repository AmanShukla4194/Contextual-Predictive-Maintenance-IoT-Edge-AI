import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PAGES = [
    "overview",
    "sensor_trends",
    "predictions",
    "maintenance_schedule",
    "alerts",
    "prediction_history",
    "model_metrics",
    "shap_analysis",
    "explainability",
    "about",
]


def test_page_imports():
    passed = []
    failed = []
    for page in PAGES:
        try:
            module = __import__(f"app.pages.{page}", fromlist=[page])
            assert hasattr(module, "render"), f"render() not found in {page}"
            passed.append(page)
        except Exception as e:
            failed.append((page, str(e)))

    print("=== Dashboard Page Import Test ===")
    for p in passed:
        print(f"  [PASS] {p}")
    for p, err in failed:
        print(f"  [FAIL] {p}: {err}")

    print(f"\n{len(passed)}/{len(PAGES)} pages passed.")
    return len(failed) == 0


def test_utils_import():
    print("\n=== Utility Module Import Test ===")
    modules = [
        "src.data.loader",
        "src.data.pipeline",
        "src.features.feature_pipeline",
        "src.models.predictor",
        "src.utils.helpers",
        "config",
    ]
    for mod in modules:
        try:
            __import__(mod)
            print(f"  [PASS] {mod}")
        except Exception as e:
            print(f"  [FAIL] {mod}: {e}")


if __name__ == "__main__":
    test_page_imports()
    test_utils_import()
