import numpy as np
import pandas as pd


def filter_by_importance(model, feature_cols, top_n=30):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    selected = [feature_cols[i] for i in indices]
    print(f"Selected top {top_n} features by model importance.")
    return selected


def importance_dataframe(model, feature_cols):
    return (
        pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def drop_zero_importance(model, feature_cols):
    imp_df = importance_dataframe(model, feature_cols)
    kept = imp_df[imp_df["importance"] > 0]["feature"].tolist()
    dropped = len(feature_cols) - len(kept)
    print(f"Dropped {dropped} zero-importance features. Kept {len(kept)}.")
    return kept
