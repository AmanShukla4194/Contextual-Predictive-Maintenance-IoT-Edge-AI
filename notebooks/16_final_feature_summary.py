import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import load_all
from src.data.pipeline import build_base_dataset
from src.features.feature_pipeline import build_features
from src.utils.helpers import get_feature_cols

if __name__ == "__main__":
    data = load_all()
    df = build_base_dataset(data)
    df = build_features(df, data["maintenance"], data["errors"])
    feature_cols = get_feature_cols(df)

    print("=== Final Feature Summary ===")
    print(f"Total features: {len(feature_cols)}")

    categories = {
        "Rolling Mean":    [c for c in feature_cols if "_mean_" in c],
        "Rolling Std":     [c for c in feature_cols if "_std_" in c],
        "Lag":             [c for c in feature_cols if "_lag_" in c],
        "Diff":            [c for c in feature_cols if "_diff_" in c],
        "Context":         [c for c in feature_cols if c in ["hours_since_maintenance", "cumulative_error_count"]],
        "Time":            [c for c in feature_cols if c in ["hour_of_day", "day_of_week", "month", "is_weekend", "week_of_year"]],
        "Machine Info":    [c for c in feature_cols if c.startswith("model_") or c == "age"],
        "Raw Sensors":     [c for c in feature_cols if c in ["volt", "rotate", "pressure", "vibration"]],
    }

    other = set(feature_cols)
    for cols in categories.values():
        other -= set(cols)
    categories["Other"] = list(other)

    print("\n--- Feature Count by Category ---")
    for cat, cols in categories.items():
        print(f"  {cat:<20}: {len(cols)}")

    print("\n--- Null Counts in Features ---")
    nulls = df[feature_cols].isnull().sum()
    null_features = nulls[nulls > 0]
    if null_features.empty:
        print("  No null values in any feature.")
    else:
        print(null_features)

    print("\n--- Feature Value Ranges (sample) ---")
    print(df[feature_cols].describe().T[["mean", "std", "min", "max"]].round(3).head(20))
