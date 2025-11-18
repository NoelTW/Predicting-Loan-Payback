"""Hyperparameter tuning with Optuna and optional training of tuned models."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score
from catboost.core import CatBoostError
from lightgbm.basic import LightGBMError
from xgboost.core import XGBoostError

from .config import MODEL_DIR, RESULTS_DIR, TRAIN_FOLDS_FILE
from .feature_engineering import build_target_encoding_state, create_features
from .inference_utils import get_prediction_probabilities
from .model_dispatcher import MODELS, get_model

TUNABLE_MODELS = {"lightgbm", "xgboost", "catboost"}

CPU_FALLBACK_PARAMS: Dict[str, Dict[str, object]] = {
    "lightgbm": {"device": "cpu"},
    "xgboost": {"device": "cpu", "tree_method": "hist"},
    "catboost": {"task_type": "CPU"},
}


FoldCache = List[Tuple[int, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Optuna to tune tree models.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=sorted(TUNABLE_MODELS),
        help="Model key from model_dispatcher to tune.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="Number of Optuna trials to execute.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Optional time budget (seconds) for the study.",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Optional Optuna study name. Defaults to '<model>_optuna'.",
    )
    parser.add_argument(
        "--train-best",
        action="store_true",
        help="Train and persist fold models using the best discovered parameters.",
    )
    parser.add_argument(
        "--model-alias",
        type=str,
        default=None,
        help=(
            "Alias to use when saving tuned models (defaults to '<model>_optuna')."
        ),
    )
    return parser.parse_args()


def _ensure_training_file() -> None:
    if not TRAIN_FOLDS_FILE.exists():
        raise FileNotFoundError(
            "train_folds.csv not found. Run `uv run python -m src.create_folds` first."
        )


def _prepare_fold_cache(df: pd.DataFrame) -> FoldCache:
    cache: FoldCache = []
    for fold in sorted(df["kfold"].unique()):
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

        cache.append((fold, X_train, y_train, X_valid, y_valid))
    return cache


def _build_model(model_name: str, params: Dict[str, object]):
    if model_name not in MODELS:
        raise KeyError(f"Model '{model_name}' is not defined in model_dispatcher.")

    model = get_model(model_name)
    if params:
        model.set_params(**params)
    return model


def _fit_model_with_fallback(model_name, model, X_train, y_train):
    try:
        model.fit(X_train, y_train)
    except (LightGBMError, XGBoostError, CatBoostError) as exc:
        fallback = CPU_FALLBACK_PARAMS.get(model_name)
        if not fallback:
            raise
        print(
            f"{model_name} GPU training failed ({exc}). Retrying on CPU for this fold."
        )
        model.set_params(**fallback)
        model.fit(X_train, y_train)


def _suggest_parameters(model_name: str, trial: optuna.trial.Trial) -> Dict[str, object]:
    if model_name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1400),
            "num_leaves": trial.suggest_int("num_leaves", 32, 256),
            "max_depth": trial.suggest_categorical("max_depth", [-1, 6, 8, 10, 12, 14]),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),
        }
    if model_name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1200),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 1e-2, 10.0, log=True
            ),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
        }
    if model_name == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 400, 1200),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.2, log=True
            ),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "bagging_temperature": trial.suggest_float(
                "bagging_temperature", 0.0, 1.0
            ),
            "border_count": trial.suggest_int("border_count", 32, 254),
        }
    raise ValueError(f"No search space configured for model '{model_name}'.")


def _evaluate_params(
    model_name: str, params: Dict[str, object], fold_cache: FoldCache
) -> float:
    scores: List[float] = []
    for fold, X_train, y_train, X_valid, y_valid in fold_cache:
        model = _build_model(model_name, params)
        _fit_model_with_fallback(model_name, model, X_train, y_train)
        preds = get_prediction_probabilities(model, X_valid)
        auc = roc_auc_score(y_valid, preds)
        scores.append(auc)
    return float(np.mean(scores))


def _objective_factory(model_name: str, fold_cache: FoldCache):
    def _objective(trial: optuna.trial.Trial) -> float:
        params = _suggest_parameters(model_name, trial)
        return _evaluate_params(model_name, params, fold_cache)

    return _objective


def _save_best_params(
    model_name: str,
    study: optuna.Study,
    trials: int,
    alias: str | None,
) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "optuna_best_params.jsonl"
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "alias": alias or f"{model_name}_optuna",
        "best_auc": study.best_value,
        "n_trials": trials,
        "params": study.best_params,
    }
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return output_path


def train_with_params(
    df: pd.DataFrame,
    model_name: str,
    alias: str,
    params: Dict[str, object],
) -> Dict[int, float]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    fold_scores: Dict[int, float] = {}

    for fold in sorted(df["kfold"].unique()):
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

        model = _build_model(model_name, params)
        _fit_model_with_fallback(model_name, model, X_train, y_train)

        preds = get_prediction_probabilities(model, X_valid)
        auc = roc_auc_score(y_valid, preds)
        fold_scores[fold] = auc

        model_path = MODEL_DIR / f"{alias}_fold{fold}.joblib"
        joblib.dump(model, model_path)
        print(
            f"Saved tuned model '{alias}' fold {fold} with AUC={auc:.4f} to {model_path.name}"
        )

    return fold_scores


def store_cv_results(
    model_name: str, alias: str, fold_scores: Dict[int, float]
) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "cv_results.csv"

    timestamp = datetime.now(timezone.utc).isoformat()
    records = [
        {
            "timestamp": timestamp,
            "model": model_name,
            "alias": alias,
            "fold": fold,
            "auc": auc,
        }
        for fold, auc in fold_scores.items()
    ]
    df_records = pd.DataFrame.from_records(records)

    if path.exists():
        existing = pd.read_csv(path)
        df_records = pd.concat([existing, df_records], ignore_index=True)

    df_records.to_csv(path, index=False)
    return path


def main() -> None:
    args = parse_args()
    _ensure_training_file()

    df = pd.read_csv(TRAIN_FOLDS_FILE)
    if "kfold" not in df.columns:
        raise ValueError("train_folds.csv must contain 'kfold' column.")

    fold_cache = _prepare_fold_cache(df)
    study_name = args.study_name or f"{args.model}_optuna"
    study = optuna.create_study(direction="maximize", study_name=study_name)
    objective = _objective_factory(args.model, fold_cache)
    study.optimize(objective, n_trials=args.trials, timeout=args.timeout)

    best_params = study.best_params
    best_auc = study.best_value
    print(f"Best CV AUC: {best_auc:.4f} with params: {best_params}")
    params_path = _save_best_params(args.model, study, args.trials, args.model_alias)
    print(f"Appended best parameters to {params_path}")

    if args.train_best:
        alias = args.model_alias or f"{args.model}_optuna"
        fold_scores = train_with_params(df, args.model, alias, best_params)
        results_path = store_cv_results(args.model, alias, fold_scores)
        mean_auc = np.mean(list(fold_scores.values()))
        std_auc = np.std(list(fold_scores.values()))
        print(
            f"Trained tuned models '{alias}' -> mean AUC {mean_auc:.4f} ± {std_auc:.4f}."
        )
        print(f"Logged fold metrics to {results_path}")
        print(
            f"Use `uv run python -m src.create_submission --models {alias}` to generate a submission."
        )


if __name__ == "__main__":
    main()
