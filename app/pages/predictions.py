import streamlit as st
import pandas as pd
from config import SENSOR_COLS


def render(model=None, feature_cols=None, label_map=None):
    st.header("Failure Predictions")

    if model is None:
        st.info("No trained model loaded. Train the model first.")
        return

    st.subheader("Enter Sensor Readings")

    col1, col2 = st.columns(2)
    inputs = {}
    for i, sensor in enumerate(SENSOR_COLS):
        with col1 if i % 2 == 0 else col2:
            inputs[sensor] = st.number_input(sensor.capitalize(), value=0.0, format="%.4f")

    if st.button("Predict Failure"):
        row = {col: 0.0 for col in feature_cols}
        for sensor in SENSOR_COLS:
            if sensor in row:
                row[sensor] = inputs[sensor]

        X = pd.DataFrame([row])[feature_cols].values
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]

        label = label_map.get(pred, str(pred)) if label_map else str(pred)
        confidence = round(float(proba.max()) * 100, 2)

        st.success(f"Predicted Class: **{label}**")
        st.info(f"Confidence: {confidence}%")

        proba_df = pd.DataFrame({
            "Class": [label_map.get(i, str(i)) for i in range(len(proba))],
            "Probability": [round(p * 100, 2) for p in proba]
        })
        st.dataframe(proba_df)
