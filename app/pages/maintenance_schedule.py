import streamlit as st
import pandas as pd


def render(telemetry_df=None, model=None, feature_cols=None, label_map=None):
    st.header("Predicted Maintenance Schedule")

    if model is None or telemetry_df is None:
        st.info("Load telemetry data and a trained model to generate the maintenance schedule.")
        return

    st.subheader("High-Risk Machines (Last 24h)")

    latest = (
        telemetry_df.sort_values("datetime")
        .groupby("machineID")
        .tail(1)
        .reset_index(drop=True)
    )

    available_cols = [c for c in feature_cols if c in latest.columns]
    missing = [c for c in feature_cols if c not in latest.columns]
    for c in missing:
        latest[c] = 0.0

    X = latest[feature_cols].values
    probas = model.predict_proba(X)
    preds = model.predict(X)

    latest["predicted_failure"] = [label_map.get(p, str(p)) if label_map else str(p) for p in preds]
    latest["confidence"] = probas.max(axis=1).round(3)

    at_risk = latest[latest["predicted_failure"] != "none"].sort_values("confidence", ascending=False)

    if at_risk.empty:
        st.success("No machines predicted to fail in the next 24 hours.")
    else:
        st.warning(f"{len(at_risk)} machine(s) flagged for maintenance.")
        display_cols = ["machineID", "predicted_failure", "confidence"]
        st.dataframe(at_risk[display_cols].reset_index(drop=True))

    st.subheader("Full Machine Risk Table")
    st.dataframe(
        latest[["machineID", "predicted_failure", "confidence"]]
        .sort_values("confidence", ascending=False)
        .reset_index(drop=True)
    )
