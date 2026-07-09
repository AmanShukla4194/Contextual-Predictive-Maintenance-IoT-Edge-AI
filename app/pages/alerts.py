import streamlit as st
import pandas as pd


def render(telemetry_df=None, model=None, feature_cols=None, label_map=None, confidence_threshold=0.75):
    st.header("Live Alerts")

    if model is None or telemetry_df is None:
        st.info("Load a trained model to generate alerts.")
        return

    threshold = st.slider("Confidence Threshold", 0.5, 0.99, confidence_threshold, step=0.01)

    latest = (
        telemetry_df.sort_values("datetime")
        .groupby("machineID")
        .tail(1)
        .reset_index(drop=True)
    )

    for c in feature_cols:
        if c not in latest.columns:
            latest[c] = 0.0

    X = latest[feature_cols].values
    preds = model.predict(X)
    probas = model.predict_proba(X)

    latest["predicted_label"] = [label_map.get(p, str(p)) if label_map else str(p) for p in preds]
    latest["confidence"] = probas.max(axis=1).round(4)

    alerts = latest[
        (latest["predicted_label"] != "none") &
        (latest["confidence"] >= threshold)
    ].sort_values("confidence", ascending=False)

    if alerts.empty:
        st.success(f"No high-confidence alerts above {threshold:.0%} threshold.")
    else:
        st.error(f"{len(alerts)} alert(s) require immediate attention.")
        for _, row in alerts.iterrows():
            st.warning(
                f"Machine **{row['machineID']}** — Predicted: **{row['predicted_label']}** "
                f"| Confidence: **{row['confidence']:.1%}**"
            )

    st.subheader("All Risk Levels")
    display = latest[["machineID", "predicted_label", "confidence"]].sort_values(
        "confidence", ascending=False
    ).reset_index(drop=True)
    st.dataframe(display, use_container_width=True)
