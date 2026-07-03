from src.data.preprocessor import merge_machine_info
from src.features.feature_pipeline import build_features
from src.models.predictor import load_model, predict, predict_proba
from src.data.loader import load_all


def run_inference(telemetry_df=None):
    data = load_all()

    if telemetry_df is None:
        telemetry_df = data["telemetry"]

    df = merge_machine_info(telemetry_df, data["machines"])
    df = build_features(df, data["maintenance"], data["errors"])

    model, feature_cols, label_map = load_model()

    missing = [c for c in feature_cols if c not in df.columns]
    for c in missing:
        df[c] = 0.0

    X = df[feature_cols].values
    preds = predict(model, X)
    probas = predict_proba(model, X)

    df = df[["machineID", "datetime"]].copy()
    df["predicted_label"] = [label_map.get(p, str(p)) for p in preds]
    df["confidence"] = probas.max(axis=1).round(4)
    return df


def run_inference_latest():
    results = run_inference()
    latest = (
        results.sort_values("datetime")
        .groupby("machineID")
        .tail(1)
        .reset_index(drop=True)
    )
    return latest
