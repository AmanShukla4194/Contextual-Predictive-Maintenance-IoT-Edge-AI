import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def evaluate(y_true, y_pred, label_map=None):
    labels = sorted(set(y_true) | set(y_pred))
    target_names = None
    if label_map:
        inv = {v: k for k, v in label_map.items()}
        target_names = [inv.get(l, str(l)) for l in labels]

    report = classification_report(
        y_true, y_pred, labels=labels, target_names=target_names, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    mf1 = macro_f1(y_true, y_pred)

    print(f"Macro F1: {mf1:.4f}\n")
    print(report)
    return {"macro_f1": mf1, "confusion_matrix": cm, "report": report}
