from src.models.predictor import load_model, predict, predict_proba, prepare_feature_frame


def run_inference(telemetry_df=None):
    model, feature_cols, label_map = load_model()
    df = prepare_feature_frame(telemetry_df, feature_cols)

    X = df[feature_cols].values
    preds = predict(model, X)
    probas = predict_proba(model, X)

    df = df[["machineID", "datetime"]].copy()
    df["predicted_label"] = preds
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
