import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import load_machines, load_failures, load_telemetry

st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")
st.title("Contextual Predictive Maintenance — IoT Edge AI")
st.markdown("---")

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Sensor Trends", "Predictions", "Explainability"])

if page == "Overview":
    st.header("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    try:
        machines = load_machines()
        failures = load_failures()
        col1.metric("Total Machines", len(machines))
        col2.metric("Total Failures", len(failures))
        col3.metric("Failure Types", failures["failure"].nunique())
        st.subheader("Machine Fleet")
        st.dataframe(machines.head(10), use_container_width=True)
        st.subheader("Failure Distribution")
        st.bar_chart(failures["failure"].value_counts())
    except Exception:
        st.warning("Dataset not found locally. Place the archive/ folder in the project root.")

elif page == "Sensor Trends":
    st.header("Sensor Telemetry Trends")
    try:
        telemetry = load_telemetry()
        machine_ids = sorted(telemetry["machineID"].unique())
        selected_machine = st.selectbox("Select Machine ID", machine_ids)
        sensor = st.selectbox("Select Sensor", ["volt", "rotate", "pressure", "vibration"])
        machine_data = telemetry[telemetry["machineID"] == selected_machine].set_index("datetime")
        st.line_chart(machine_data[sensor])
    except Exception:
        st.warning("Dataset not found locally. Place the archive/ folder in the project root.")

elif page == "Predictions":
    st.header("Failure Prediction Interface")
    st.markdown("Enter sensor readings below to predict component failure risk.")
    col1, col2 = st.columns(2)
    with col1:
        volt = st.number_input("Voltage (volt)", min_value=0.0, value=170.0, step=0.1)
        rotate = st.number_input("Rotation Speed (rotate)", min_value=0.0, value=450.0, step=0.1)
    with col2:
        pressure = st.number_input("Pressure (pressure)", min_value=0.0, value=100.0, step=0.1)
        vibration = st.number_input("Vibration (vibration)", min_value=0.0, value=40.0, step=0.1)
    if st.button("Predict Failure Risk"):
        st.info("Model not yet integrated. Prediction will be available after training is complete.")

elif page == "Explainability":
    st.header("SHAP Feature Importance")
    st.write("Model explainability module — coming soon.")
