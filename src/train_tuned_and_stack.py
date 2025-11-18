"""Train models using saved Optuna params and build a stacked submission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

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
from .create_stacking_submission import build_oof_and_test_predictions
from .inference_utils import (
    discover_model_paths,
    prepare_fold_feature_metadata,
    validate_selected_models,
)
from .optuna_tuner import store_cv_results, train_with_params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train tuned models from Optuna params and create a stacked submission."
    )
    parser.add_argument(
        "--params-file",
        type=Path,
        default=RESULTS_DIR / "optuna_best_params.jsonl",
        help="Path to JSONL file with Optuna best params records.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional subset of aliases or base model names to train/stack.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "submission_tuned_stack.csv",
        help="Path for the stacked submission CSV.",
    )
    parser.add_argument(
        "--oof-output",
        type=Path,
        default=RESULTS_DIR / "tuned_stacking_oof.csv",
        help="Where to save OOF predictions for the base models.",
    )
    return parser.parse_args()


def load_best_param_records(
    params_path: Path, requested: List[str] | None
) -> Dict[str, Dict[str, object]]:
    if not params_path.exists():
        raise FileNotFoundError(
            f"Params file '{params_path}' not found. Run optuna tuning first."
        )

    records: Dict[str, Dict[str, object]] = {}
    by_model: Dict[str, str] = {}

    with params_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            alias = payload.get("alias") or f"{payload['model']}_optuna"
            payload["alias"] = alias
            records[alias] = payload
            by_model[payload["model"]] = alias

    if not records:
        raise RuntimeError(f"No parameter records found in {params_path}.")

    if not requested:
        return records

    selected: Dict[str, Dict[str, object]] = {}
    missing: List[str] = []
    for name in requested:
        if name in records:
            selected[name] = records[name]
            continue
        if name in by_model:
            alias = by_model[name]
            selected[alias] = records[alias]
            continue
        missing.append(name)

    if missing:
        raise ValueError(
            f"Requested models/aliases {missing} not found in {params_path}."
        )
    return selected


def train_models_from_records(
    train_df: pd.DataFrame, records: Dict[str, Dict[str, object]]
) -> None:
    for alias, payload in records.items():
        params = payload["params"]
        model_name = payload["model"]
        print(f"Training '{alias}' ({model_name}) with tuned parameters.")
        fold_scores = train_with_params(train_df, model_name, alias, params)
        store_cv_results(model_name, alias, fold_scores)
        print(
            f"Finished '{alias}' -> mean AUC {np.mean(list(fold_scores.values())):.4f}"
        )


def run_stacking_for_aliases(
    aliases: List[str], output: Path, oof_output: Path
) -> None:
    discovered = discover_model_paths()
    selected_models = validate_selected_models(aliases, discovered)
    if not selected_models:
        raise RuntimeError(
            f"No trained models found matching aliases {aliases}. Training step may have failed."
        )

    train_df = pd.read_csv(TRAIN_FOLDS_FILE)
    fold_ids = sorted({fold for folds in selected_models.values() for fold in folds})
    fold_metadata = prepare_fold_feature_metadata(train_df, fold_ids)
    test_df = pd.read_csv(TEST_FILE)

    oof_df, test_meta = build_oof_and_test_predictions(
        train_df, test_df, selected_models, fold_metadata
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    oof_df.to_csv(oof_output, index=False)
    print(f"Saved OOF base predictions to {oof_output}")

    feature_cols = aliases
    meta_X = oof_df[feature_cols].values
    meta_y = oof_df[TARGET_COL].values

    meta_model = LogisticRegression(max_iter=2000)
    meta_model.fit(meta_X, meta_y)
    oof_meta_pred = meta_model.predict_proba(meta_X)[:, 1]
    oof_auc = roc_auc_score(meta_y, oof_meta_pred)
    print(f"Meta-model OOF AUC: {oof_auc:.4f}")

    test_meta_X = test_meta[feature_cols].values
    stacked_preds = meta_model.predict_proba(test_meta_X)[:, 1]
    submission = pd.DataFrame(
        {
            ID_COL: test_df[ID_COL],
            TARGET_COL: stacked_preds,
        }
    )
    submission.to_csv(output, index=False)
    print(f"Wrote stacked submission with tuned models to {output}")


def main() -> None:
    args = parse_args()

    if not TRAIN_FOLDS_FILE.exists():
        raise FileNotFoundError(
            "Missing processed training folds. Run `uv run python -m src.create_folds` first."
        )
    if not TEST_FILE.exists():
        raise FileNotFoundError("Missing test.csv in raw data directory.")

    train_df = pd.read_csv(TRAIN_FOLDS_FILE)
    records = load_best_param_records(args.params_file, args.models)
    aliases = list(records.keys())
    train_models_from_records(train_df, records)
    run_stacking_for_aliases(aliases, args.output, args.oof_output)


if __name__ == "__main__":
    main()
