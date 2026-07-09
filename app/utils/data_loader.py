import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
from src.data.loader import load_all
from src.models.predictor import load_model


@st.cache_data
def get_telemetry():
    data = load_all()
    return data["telemetry"], data["failures"], data["machines"]


@st.cache_resource
def get_model():
    try:
        model, feature_cols, label_map = load_model()
        return model, feature_cols, label_map
    except FileNotFoundError:
        return None, None, None
