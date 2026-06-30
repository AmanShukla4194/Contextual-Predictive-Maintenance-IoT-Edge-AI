import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.data_loader import get_telemetry, get_model
from app.pages import (
    overview, sensor_trends, predictions, explainability,
    maintenance_schedule, alerts, prediction_history, model_metrics,
    shap_analysis, about
)

st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")
st.title("Contextual Predictive Maintenance — IoT Edge AI")
st.markdown("---")

telemetry_df, failures_df, machines_df = get_telemetry()
model, feature_cols, label_map = get_model()

st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Overview", "Sensor Trends", "Predictions",
        "Maintenance Schedule", "Alerts", "Prediction History",
        "Model Metrics", "SHAP Analysis", "Explainability", "About"
    ]
)

if page == "Overview":
    overview.render(telemetry_df, failures_df, machines_df)
elif page == "Sensor Trends":
    sensor_trends.render(telemetry_df)
elif page == "Predictions":
    predictions.render(model, feature_cols, label_map)
elif page == "Maintenance Schedule":
    maintenance_schedule.render(telemetry_df, model, feature_cols, label_map)
elif page == "Alerts":
    alerts.render(telemetry_df, model, feature_cols, label_map)
elif page == "Prediction History":
    prediction_history.render()
elif page == "Model Metrics":
    model_metrics.render()
elif page == "SHAP Analysis":
    shap_analysis.render(model, feature_cols, telemetry_df)
elif page == "Explainability":
    explainability.render(model, feature_cols)
elif page == "About":
    about.render()
