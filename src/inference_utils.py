"""Shared helpers for inference/submission scripts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, Mapping

import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.special import expit

from .config import MODEL_DIR
from .feature_engineering import build_target_encoding_state, create_features

MODEL_FILENAME_PATTERN = re.compile(r"(?P<name>.+)_fold(?P<fold>\d+)\.joblib$")


def get_prediction_probabilities(model, X_valid: DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X_valid))
        return proba[:, 1] if proba.ndim == 2 else proba.ravel()
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X_valid))
        return expit(scores)
    preds = np.asarray(model.predict(X_valid))
    return preds if preds.ndim == 1 else preds[:, 0]


def discover_model_paths() -> Dict[str, Dict[int, Path]]:
    discovered: Dict[str, Dict[int, Path]] = {}
    for path in MODEL_DIR.glob("*_fold*.joblib"):
        match = MODEL_FILENAME_PATTERN.match(path.name)
        if not match:
            continue
        name = match.group("name")
        fold = int(match.group("fold"))
        discovered.setdefault(name, {})[fold] = path
    return discovered


def validate_selected_models(
    requested: Iterable[str] | None, discovered: Mapping[str, Dict[int, Path]]
) -> Dict[str, Dict[int, Path]]:
    if not requested:
        return dict(discovered)

    requested = list(requested)
    missing = [name for name in requested if name not in discovered]
    if missing:
        raise ValueError(f"Requested models missing trained artifacts: {missing}")

    return {name: discovered[name] for name in requested}


def prepare_fold_feature_metadata(
    train_df: DataFrame, folds: Iterable[int]
) -> Dict[int, Dict[str, object]]:
    metadata: Dict[int, Dict[str, object]] = {}
    for fold in folds:
        train_subset = train_df[train_df["kfold"] != fold].reset_index(drop=True)
        state, target_mean = build_target_encoding_state(train_subset)
        X_train, _ = create_features(
            train_subset,
            is_train=True,
            target_encoding_state=state,
            target_encoding_mean=target_mean,
            fit_target_encoding=True,
        )
        metadata[fold] = {
            "columns": list(X_train.columns),
            "target_state": state,
            "target_mean": target_mean,
        }
    return metadata


__all__ = [
    "MODEL_FILENAME_PATTERN",
    "prepare_fold_feature_metadata",
    "discover_model_paths",
    "get_prediction_probabilities",
    "validate_selected_models",
]
