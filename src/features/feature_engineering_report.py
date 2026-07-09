import os
import json
from config import REPORTS_DIR


FEATURE_CATEGORIES = {
    "raw_sensors":       lambda c: c in ["volt", "rotate", "pressure", "vibration"],
    "rolling_mean":      lambda c: "_mean_" in c,
    "rolling_std":       lambda c: "_std_" in c,
    "rolling_min":       lambda c: "_min_" in c,
    "rolling_max":       lambda c: "_max_" in c,
    "rolling_range":     lambda c: "_range_" in c,
    "lag":               lambda c: "_lag_" in c,
    "diff":              lambda c: "_diff_" in c,
    "ewm":               lambda c: "_ewm_" in c,
    "smoothing_sma":     lambda c: "_sma_" in c,
    "anomaly_zscore":    lambda c: "_zscore" in c,
    "anomaly_flag":      lambda c: c == "anomaly_flag",
    "anomaly_score":     lambda c: c == "anomaly_score",
    "spike_magnitude":   lambda c: "_spike" in c and "_flag" not in c,
    "spike_flag":        lambda c: "_spike_flag" in c,
    "context":           lambda c: c in ["hours_since_maintenance", "cumulative_error_count"],
    "time":              lambda c: c in ["hour_of_day", "day_of_week", "month", "is_weekend", "week_of_year"],
    "machine_info":      lambda c: c.startswith("model_") or c == "age",
}


def categorise_features(feature_cols):
    report = {cat: [] for cat in FEATURE_CATEGORIES}
    report["uncategorised"] = []
    for col in feature_cols:
        matched = False
        for cat, fn in FEATURE_CATEGORIES.items():
            if fn(col):
                report[cat].append(col)
                matched = True
                break
        if not matched:
            report["uncategorised"].append(col)
    return report


def generate_feature_report(feature_cols):
    report = categorise_features(feature_cols)
    summary = {cat: len(cols) for cat, cols in report.items()}
    summary["total"] = len(feature_cols)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    out_path = os.path.join(REPORTS_DIR, "feature_engineering_report.json")
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "features_by_category": report}, f, indent=2)

    print("=== Feature Engineering Report ===")
    for cat, count in summary.items():
        if cat != "total":
            print(f"  {cat:<25}: {count}")
    print(f"  {'TOTAL':<25}: {summary['total']}")
    print(f"\nReport saved to {out_path}")
    return summary
