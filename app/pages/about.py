import streamlit as st


def render():
    st.header("About This Project")

    st.subheader("Contextual Predictive Maintenance — IoT Edge AI")
    st.markdown("""
    This system predicts industrial equipment failures up to **24 hours in advance**
    using sensor data from IoT devices. It combines time-series feature engineering
    with a LightGBM classifier trained on the Microsoft Azure Predictive Maintenance dataset.
    """)

    st.subheader("Dataset")
    st.markdown("""
    **Microsoft Azure PdM Dataset**
    - 100 machines monitored over 1 year
    - 876,000+ hourly telemetry records
    - 4 sensors: voltage, rotation, pressure, vibration
    - 5 failure types: none, comp1, comp2, comp3, comp4
    """)

    st.subheader("ML Approach")
    st.markdown("""
    - **Model**: LightGBM multi-class classifier
    - **Imbalance handling**: SMOTE inside stratified K-fold CV
    - **Evaluation metric**: Macro F1 Score
    - **Features**: Rolling stats, lag, diff, context, time, anomaly scores
    - **Explainability**: SHAP values
    """)

    st.subheader("Team")
    data = {
        "Name": ["Aman Shukla", "Gokul", "Nakshatra", "Vrithik"],
        "Role": ["Team Leader — ML Pipeline", "Data Pipeline & EDA", "Feature Engineering", "Dashboard & Visualisation"],
        "Branch": ["aman-shukla", "gokul", "nakshatra", "vrithik"],
    }
    st.table(data)

    st.subheader("Tech Stack")
    st.markdown("""
    Python · LightGBM · scikit-learn · imbalanced-learn · SHAP · Streamlit · Plotly · pandas · joblib
    """)
