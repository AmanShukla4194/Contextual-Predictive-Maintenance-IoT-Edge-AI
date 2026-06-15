import pandas as pd
import numpy as np
from config import FAILURE_COMPONENTS, RANDOM_STATE


def encode_labels(df):
    label_map = {"none": 0}
    for i, comp in enumerate(FAILURE_COMPONENTS, start=1):
        label_map[comp] = i
    df = df.copy()
    df["label_enc"] = df["label"].map(label_map).fillna(0).astype(int)
    return df, label_map


def split_by_time(df, test_months=2):
    cutoff = df["datetime"].max() - pd.DateOffset(months=test_months)
    train = df[df["datetime"] <= cutoff].reset_index(drop=True)
    test = df[df["datetime"] > cutoff].reset_index(drop=True)
    return train, test


def get_feature_cols(df, exclude=None):
    if exclude is None:
        exclude = ["machineID", "datetime", "label", "label_enc"]
    return [c for c in df.columns if c not in exclude]
