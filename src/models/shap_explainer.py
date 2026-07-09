import numpy as np
import shap


def compute_shap_values(model, X, max_samples=500):
    if len(X) > max_samples:
        idx = np.random.choice(len(X), max_samples, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    return explainer, shap_values, X_sample


def top_features_by_shap(shap_values, feature_names, top_n=15):
    if isinstance(shap_values, list):
        mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        mean_abs = np.abs(shap_values).mean(axis=0)
    importance = dict(zip(feature_names, mean_abs))
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    return sorted_features[:top_n]
