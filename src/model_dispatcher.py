"""Central place to instantiate ML models."""

from __future__ import annotations

from typing import Dict

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

MODELS: Dict[str, object] = {
    "log_reg": LogisticRegression(
        max_iter=2000,
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    ),
    "gradient_boosting": GradientBoostingClassifier(
        random_state=42,
    ),
    "xgboost": XGBClassifier(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        tree_method="hist",
        device="cuda",
        random_state=42,
    ),
    "lightgbm": LGBMClassifier(
        n_estimators=800,
        max_depth=-1,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary",
        random_state=42,
        device="gpu",
        boosting_type="gbdt",
    ),
    "catboost": CatBoostClassifier(
        iterations=800,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=False,
        task_type="GPU",
        devices="0",
    ),
}


def get_model(name: str):
    """Return a cloned model instance for ``name``."""
    try:
        model = MODELS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown model '{name}'. Available: {list(MODELS)}") from exc

    from sklearn.base import clone
    from sklearn.base import BaseEstimator

    if isinstance(model, BaseEstimator):
        return clone(model)

    params = model.get_params() if hasattr(model, "get_params") else {}
    return model.__class__(**params)


__all__ = ["MODELS", "get_model"]
