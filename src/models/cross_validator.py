import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from config import RANDOM_STATE


def run_cv(X, y, params, n_splits=5):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        smote = SMOTE(random_state=RANDOM_STATE)
        X_res, y_res = smote.fit_resample(X_train, y_train)

        model = LGBMClassifier(class_weight="balanced", random_state=RANDOM_STATE, **params)
        model.fit(X_res, y_res)

        preds = model.predict(X_val)
        score = f1_score(y_val, preds, average="macro", zero_division=0)
        fold_scores.append(score)
        print(f"  Fold {fold}: Macro F1 = {score:.4f}")

    mean_f1 = np.mean(fold_scores)
    std_f1 = np.std(fold_scores)
    print(f"\nCV Result: {mean_f1:.4f} +/- {std_f1:.4f}")
    return fold_scores, mean_f1, std_f1
