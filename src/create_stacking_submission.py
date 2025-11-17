"""Stack base model predictions using a meta-model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .config import (
    BASE_DIR,
    ID_COL,
    RESULTS_DIR,
    TARGET_COL,
    TEST_FILE,
    TRAIN_FOLDS_FILE,
)
from .feature_engineering import create_features
from .inference_utils import (
    compute_fold_feature_columns,
    discover_model_paths,
    get_prediction_probabilities,
    validate_selected_models,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create stacked submission using saved base models.")
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Model names to include (defaults to all discovered models).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "stacked_submission.csv",
        help="Path to write stacked submission CSV.",
    )
    parser.add_argument(
        "--oof-output",
        type=Path,
        default=RESULTS_DIR / "stacking_oof.csv",
        help="Where to store out-of-fold base predictions.",
    )
    return parser.parse_args()


def build_oof_and_test_predictions(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    selected_models: Dict[str, Dict[int, Path]],
    fold_columns: Dict[int, List[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    oof_df = train_df[[ID_COL, TARGET_COL, "kfold"]].copy()
    test_meta = pd.DataFrame({ID_COL: test_df[ID_COL]})
    X_test_base, _ = create_features(test_df, is_train=False)

    for model_name in selected_models:
        oof_df[model_name] = np.nan

    for model_name in selected_models:
        fold_test_preds: List[np.ndarray] = []
        for fold, model_path in sorted(selected_models[model_name].items()):
            model = joblib.load(model_path)
            expected_cols = fold_columns[fold]

            valid_mask = train_df["kfold"] == fold
            valid_subset = train_df.loc[valid_mask]
            X_valid, _ = create_features(valid_subset, is_train=True)
            X_valid = X_valid.reindex(columns=expected_cols, fill_value=0)

            preds_valid = get_prediction_probabilities(model, X_valid)
            oof_df.loc[valid_mask, model_name] = preds_valid

            X_test = X_test_base.reindex(columns=expected_cols, fill_value=0)
            preds_test = get_prediction_probabilities(model, X_test)
            fold_test_preds.append(preds_test)

        if not fold_test_preds:
            raise RuntimeError(f"No trained folds found for model '{model_name}'.")

        test_meta[model_name] = np.mean(fold_test_preds, axis=0)

        if oof_df[model_name].isna().any():
            raise RuntimeError(f"OOF predictions missing for model '{model_name}'.")

    return oof_df, test_meta


def main() -> None:
    args = parse_args()

    if not TRAIN_FOLDS_FILE.exists():
        raise FileNotFoundError("Missing train_folds.csv. Run `uv run python -m src.create_folds` first.")
    if not TEST_FILE.exists():
        raise FileNotFoundError("Missing test.csv in raw data directory.")

    discovered = discover_model_paths()
    if not discovered:
        raise FileNotFoundError("No models found in models/. Train base models first.")

    selected_models = validate_selected_models(args.models, discovered)
    all_folds = sorted({fold for folds in selected_models.values() for fold in folds})

    train_df = pd.read_csv(TRAIN_FOLDS_FILE)
    fold_columns = compute_fold_feature_columns(train_df, all_folds)
    test_df = pd.read_csv(TEST_FILE)

    oof_df, test_meta = build_oof_and_test_predictions(train_df, test_df, selected_models, fold_columns)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    oof_df.to_csv(args.oof_output, index=False)
    print(f"Saved out-of-fold predictions to {args.oof_output}")

    feature_cols = list(selected_models.keys())
    meta_X = oof_df[feature_cols].values
    meta_y = oof_df[TARGET_COL].values

    meta_model = LogisticRegression(max_iter=2000)
    meta_model.fit(meta_X, meta_y)
    oof_meta_preds = meta_model.predict_proba(meta_X)[:, 1]
    oof_auc = roc_auc_score(meta_y, oof_meta_preds)
    print(f"Meta-model training AUC on OOF features: {oof_auc:.4f}")

    test_meta_X = test_meta[feature_cols].values
    stacked_preds = meta_model.predict_proba(test_meta_X)[:, 1]

    submission_df = pd.DataFrame(
        {
            ID_COL: test_df[ID_COL],
            TARGET_COL: stacked_preds,
        }
    )
    submission_df.to_csv(args.output, index=False)
    print(f"Wrote stacked submission to {args.output}")


if __name__ == "__main__":
    main()
