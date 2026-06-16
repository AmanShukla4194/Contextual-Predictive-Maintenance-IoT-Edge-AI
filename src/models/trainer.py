import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from config import RANDOM_STATE
from src.models.evaluator import macro_f1


def train_lgbm(X_train, y_train, params=None):
    if params is None:
        params = {
            "objective": "multiclass",
            "num_class": 5,
            "metric": "multi_logloss",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_child_samples": 20,
            "random_state": RANDOM_STATE,
            "verbose": -1,
        }
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    return model


def cross_validate(X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        smote = SMOTE(random_state=RANDOM_STATE)
        X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)
        model = train_lgbm(X_tr_res, y_tr_res)
        preds = model.predict(X_val)
        score = macro_f1(y_val, preds)
        scores.append(score)
        print(f"Fold {fold} Macro F1: {score:.4f}")
    print(f"\nMean Macro F1: {np.mean(scores):.4f} +/- {np.std(scores):.4f}")
    return scores
