import numpy as np
from sklearn.metrics import f1_score


def tune_threshold(y_true, y_proba, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.1, 0.9, 0.05)

    best_thresh = 0.5
    best_f1 = 0.0

    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        score = f1_score(y_true, preds, average="macro", zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = t

    print(f"Best threshold: {best_thresh:.2f} | Macro F1: {best_f1:.4f}")
    return best_thresh, best_f1


def apply_threshold(y_proba, threshold):
    return (y_proba >= threshold).astype(int)
