import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


def render(model=None, feature_cols=None):
    st.header("Model Explainability")

    if model is None or feature_cols is None:
        st.info("Train and load a model to view SHAP-based explanations.")
        return

    st.subheader("Feature Importance (Built-in)")
    importance = model.feature_importances_
    imp_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": importance
    }).sort_values("Importance", ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(imp_df["Feature"][::-1], imp_df["Importance"][::-1], color="steelblue")
    ax.set_xlabel("Importance Score")
    ax.set_title("Top 20 Features by Importance")
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Top Features Table")
    st.dataframe(imp_df.reset_index(drop=True))
