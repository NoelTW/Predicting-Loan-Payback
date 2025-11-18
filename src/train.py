"""Training script with cross-validation and AUC tracking."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score

from .config import MODEL_DIR, NUM_FOLDS, RESULTS_DIR, TARGET_COL, TRAIN_FOLDS_FILE
from .feature_engineering import build_target_encoding_state, create_features
from .model_dispatcher import MODELS, get_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train models with CV and store artifacts.")
    parser.add_argument(
        "--model",
        type=str,
        default="log_reg",
        choices=sorted(MODELS.keys()),
        help="Model key defined in model_dispatcher.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Optional fold number to train. Default trains all folds.",
    )
    return parser.parse_args()


def _ensure_data_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Please run `uv run python -m src.create_folds` first."
        )


def _get_prediction_probabilities(model, X_valid) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_valid)
        proba = np.asarray(proba)
        return proba[:, 1] if proba.ndim == 2 else proba.ravel()
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_valid)
        return expit(np.asarray(scores))
    preds = model.predict(X_valid)
    preds = np.asarray(preds)
    return preds if preds.ndim == 1 else preds[:, 0]


def _train_single_fold(df: pd.DataFrame, model_name: str, fold: int) -> float:
    train_df = df[df["kfold"] != fold].reset_index(drop=True)
    valid_df = df[df["kfold"] == fold].reset_index(drop=True)

    target_state, target_mean = build_target_encoding_state(train_df)
    X_train, y_train = create_features(
        train_df,
        is_train=True,
        target_encoding_state=target_state,
        target_encoding_mean=target_mean,
        fit_target_encoding=True,
    )
    X_valid, y_valid = create_features(
        valid_df,
        is_train=True,
        target_encoding_state=target_state,
        target_encoding_mean=target_mean,
        fit_target_encoding=False,
    )

    model = get_model(model_name)
    model.fit(X_train, y_train)

    preds = _get_prediction_probabilities(model, X_valid)
    auc = roc_auc_score(y_valid, preds)

    model_path = MODEL_DIR / f"{model_name}_fold{fold}.joblib"
    joblib.dump(model, model_path)

    print(f"Fold {fold}: AUC = {auc:.4f} (saved {model_path.name})")
    return auc


def _store_cv_results(model_name: str, fold_scores: Dict[int, float]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "cv_results.csv"

    timestamp = datetime.now(timezone.utc).isoformat()
    records = [
        {
            "timestamp": timestamp,
            "model": model_name,
            "fold": fold_id,
            "auc": auc,
        }
        for fold_id, auc in fold_scores.items()
    ]
    df_records = pd.DataFrame.from_records(records)

    if results_path.exists():
        existing = pd.read_csv(results_path)
        df_records = pd.concat([existing, df_records], ignore_index=True)

    df_records.to_csv(results_path, index=False)
    return results_path


def run(model_name: str, fold: int | None = None) -> None:
    _ensure_data_file(TRAIN_FOLDS_FILE)
    df = pd.read_csv(TRAIN_FOLDS_FILE)

    for required_col in ("kfold", TARGET_COL):
        if required_col not in df.columns:
            raise ValueError(f"train_folds.csv must contain '{required_col}'.")

    folds: Iterable[int]
    if fold is not None:
        folds = [fold]
    else:
        folds = sorted(df["kfold"].unique())

    fold_scores: Dict[int, float] = {}
    for fold_id in folds:
        fold_scores[fold_id] = _train_single_fold(df, model_name, fold_id)

    scores = list(fold_scores.values())
    results_path = _store_cv_results(model_name, fold_scores)

    if len(scores) > 1:
        mean_auc = np.mean(scores)
        std_auc = np.std(scores)
        print(f"Overall AUC: {mean_auc:.4f} ± {std_auc:.4f} (appended to {results_path})")
    else:
        only_fold = next(iter(fold_scores))
        print(f"Completed fold {only_fold}: AUC = {scores[0]:.4f} (appended to {results_path})")


def main() -> None:
    args = parse_args()
    if args.fold is not None and not (0 <= args.fold < NUM_FOLDS):
        raise ValueError(f"Fold must be between 0 and {NUM_FOLDS - 1}, got {args.fold}.")
    run(args.model, args.fold)


if __name__ == "__main__":
    main()
