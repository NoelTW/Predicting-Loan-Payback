"""Generate competition submission by ensembling trained models."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd

from .config import (
    BASE_DIR,
    ID_COL,
    MODEL_DIR,
    TARGET_COL,
    TEST_FILE,
    TRAIN_FOLDS_FILE,
)
from .feature_engineering import create_features
from .inference_utils import (
    discover_model_paths,
    get_prediction_probabilities,
    prepare_fold_feature_metadata,
    validate_selected_models,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Kaggle submission from saved models.")
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Model names to include (defaults to all discovered models).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "submission.csv",
        help="Path to write submission CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not TRAIN_FOLDS_FILE.exists():
        raise FileNotFoundError("Missing train_folds.csv. Run `uv run python -m src.create_folds` first.")
    if not TEST_FILE.exists():
        raise FileNotFoundError("Missing test.csv in raw data directory.")

    discovered = discover_model_paths()
    if not discovered:
        raise FileNotFoundError(f"No models found in {MODEL_DIR}. Train models before creating submission.")

    selected_models = validate_selected_models(args.models, discovered)
    all_folds = sorted({fold for folds in selected_models.values() for fold in folds})

    train_df = pd.read_csv(TRAIN_FOLDS_FILE)
    fold_metadata = prepare_fold_feature_metadata(train_df, all_folds)

    test_df = pd.read_csv(TEST_FILE)
    test_ids = test_df[ID_COL].copy()
    test_feature_cache: Dict[int, pd.DataFrame] = {}

    ensemble_pred = np.zeros(len(test_ids), dtype=float)
    model_count = 0

    for model_name, folds in selected_models.items():
        for fold, model_path in sorted(folds.items()):
            model = joblib.load(model_path)
            metadata = fold_metadata[fold]
            expected_columns = metadata["columns"]
            if fold not in test_feature_cache:
                X_test_fold, _ = create_features(
                    test_df,
                    is_train=False,
                    target_encoding_state=metadata["target_state"],
                    target_encoding_mean=metadata["target_mean"],
                )
                test_feature_cache[fold] = X_test_fold
            X_test = test_feature_cache[fold].reindex(columns=expected_columns, fill_value=0)
            preds = get_prediction_probabilities(model, X_test)
            ensemble_pred += preds
            model_count += 1
            print(f"Inferred with {model_name} fold {fold} -> mean prob {preds.mean():.4f}")

    if model_count == 0:
        raise RuntimeError("No models processed; cannot create submission.")

    final_predictions = ensemble_pred / model_count

    submission_df = pd.DataFrame(
        {
            ID_COL: test_ids,
            TARGET_COL: final_predictions,
        }
    )
    submission_df.to_csv(args.output, index=False)
    print(f"Wrote submission for {model_count} models to {args.output}")


if __name__ == "__main__":
    main()
