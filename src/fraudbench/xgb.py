"""XGBoost baseline on node features only."""

import numpy as np
from xgboost import XGBClassifier

DEFAULT_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "tree_method": "hist",
    "eval_metric": "aucpr",
    "n_jobs": -1,
    "random_state": 42,
}


def train_xgboost(x_train: np.ndarray, y_train: np.ndarray, params: dict | None = None) -> tuple[XGBClassifier, dict]:
    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    # Class imbalance: weight positives by negatives/positives, from training labels only.
    cfg = {**DEFAULT_PARAMS, **(params or {}), "scale_pos_weight": n_neg / n_pos}
    model = XGBClassifier(**cfg)
    model.fit(x_train, y_train)
    return model, cfg
