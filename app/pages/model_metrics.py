import streamlit as st
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import REPORTS_DIR


def render():
    st.header("Model Evaluation Metrics")

    report_path = os.path.join(REPORTS_DIR, "evaluation_results.txt")
    json_path = os.path.join(REPORTS_DIR, "evaluation_summary.json")

    if not os.path.exists(report_path):
        st.info("No evaluation report found. Run the evaluation pipeline first.")
        return

    with open(report_path, "r") as f:
        report_text = f.read()

    for line in report_text.splitlines():
        if line.startswith("Macro F1:"):
            f1_val = float(line.split(":")[1].strip())
            st.metric("Macro F1 Score", f"{f1_val:.4f}")
            break

    st.subheader("Classification Report")
    st.code(report_text, language="text")

    if os.path.exists(json_path):
        import json
        with open(json_path, "r") as f:
            summary = json.load(f)
        st.subheader("Summary JSON")
        st.json(summary)
