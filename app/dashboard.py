import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")

st.title("Contextual Predictive Maintenance — IoT Edge AI")
st.markdown("---")

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Predictions", "Explainability"])

if page == "Overview":
    st.header("Project Overview")
    st.write(
        "This dashboard demonstrates the predictive maintenance pipeline "
        "built on the Microsoft Azure PdM dataset."
    )
    st.info("Use the sidebar to navigate between sections.")

elif page == "Predictions":
    st.header("Failure Predictions")
    st.write("Live prediction interface — coming soon.")

elif page == "Explainability":
    st.header("SHAP Feature Importance")
    st.write("Model explainability module — coming soon.")
