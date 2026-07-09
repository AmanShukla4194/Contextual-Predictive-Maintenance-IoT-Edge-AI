import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def render(model=None, feature_cols=None, sample_df=None):
    st.header("SHAP Explainability Analysis")

    if model is None or feature_cols is None:
        st.info("Load a trained model to view SHAP explanations.")
        return

    try:
        import shap
    except ImportError:
        st.error("SHAP library not installed. Run: pip install shap")
        return

    if sample_df is None or sample_df.empty:
        st.warning("No sample data available for SHAP analysis.")
        return

    st.subheader("Global Feature Importance (SHAP)")
    sample_size = min(500, len(sample_df))
    X_sample = sample_df[feature_cols].values[:sample_size]

    with st.spinner("Computing SHAP values..."):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        mean_shap = np.abs(shap_values).mean(axis=0)

    top_n = 20
    top_idx = np.argsort(mean_shap)[::-1][:top_n]
    top_features = [feature_cols[i] for i in top_idx]
    top_values = mean_shap[top_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_features[::-1], top_values[::-1], color="steelblue")
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title(f"Top {top_n} Features by SHAP Importance")
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("SHAP Feature Table")
    shap_df = pd.DataFrame({"Feature": top_features, "Mean |SHAP|": top_values.round(5)})
    st.dataframe(shap_df.reset_index(drop=True), use_container_width=True)
