import streamlit as st
import pandas as pd
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.prediction_logger import read_log, LOG_FILE


def render():
    st.header("Prediction History")

    rows = read_log()
    if not rows:
        st.info("No predictions logged yet. Run predictions from the Predictions page.")
        return

    df = pd.DataFrame(rows)
    df["confidence"] = df["confidence"].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Predictions Logged", len(df))
    col2.metric("Unique Machines", df["machineID"].nunique())
    flagged = (df["predicted_label"] != "none").sum()
    col3.metric("Failures Predicted", int(flagged))

    st.subheader("Recent Predictions")
    st.dataframe(
        df.sort_values("timestamp", ascending=False).head(50).reset_index(drop=True),
        use_container_width=True
    )

    st.subheader("Prediction Label Breakdown")
    label_counts = df["predicted_label"].value_counts().reset_index()
    label_counts.columns = ["Label", "Count"]
    st.bar_chart(label_counts.set_index("Label"))
