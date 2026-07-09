import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from src.models.predictor import inverse_label_map


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def evaluate(y_true, y_pred, label_map=None):
    labels = sorted(set(y_true) | set(y_pred))
    target_names = None
    if label_map:
        inv = inverse_label_map(label_map)
        target_names = [inv.get(l, str(l)) for l in labels]

    report = classification_report(
        y_true, y_pred, labels=labels, target_names=target_names, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    mf1 = macro_f1(y_true, y_pred)
    per_class_scores = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class_names = target_names or [str(l) for l in labels]
    per_class_f1 = {
        name: float(score)
        for name, score in zip(per_class_names, per_class_scores)
    }

    print(f"Macro F1: {mf1:.4f}\n")
    print(report)
    return {
        "macro_f1": mf1,
        "confusion_matrix": cm,
        "report": report,
        "per_class_f1": per_class_f1,
    }
