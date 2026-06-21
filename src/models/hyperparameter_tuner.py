import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from config import RANDOM_STATE


PARAM_GRID = [
    {"n_estimators": 200, "max_depth": 6,  "learning_rate": 0.05, "num_leaves": 31},
    {"n_estimators": 300, "max_depth": 8,  "learning_rate": 0.03, "num_leaves": 63},
    {"n_estimators": 150, "max_depth": 5,  "learning_rate": 0.1,  "num_leaves": 20},
    {"n_estimators": 400, "max_depth": 10, "learning_rate": 0.02, "num_leaves": 127},
]


def search_best_params(X, y, param_grid=None, cv_folds=5):
    if param_grid is None:
        param_grid = PARAM_GRID

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    best_score = -np.inf
    best_params = None

    for params in param_grid:
        model = LGBMClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
            **params
        )
        scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
        mean_score = scores.mean()
        print(f"Params: {params} | Macro F1: {mean_score:.4f} (+/- {scores.std():.4f})")

        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    print(f"\nBest Macro F1: {best_score:.4f}")
    print(f"Best Params: {best_params}")
    return best_params, best_score
