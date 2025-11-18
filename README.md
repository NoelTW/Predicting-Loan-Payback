# Predicting-Loan-Payback

Overview
Welcome to the 2025 Kaggle Playground Series! We plan to continue in the spirit of previous playgrounds, providing interesting and approachable datasets for our community to practice their machine learning skills, and anticipate a competition each month.

Your Goal: Predict the probability that a borrower will pay back their loan.

## Training, Optuna tuning, and stacked submissions

You can explore more powerful ensembles by running Optuna directly from this repo:

1. Generate the stratified folds once:
   ```bash
   uv run python -m src.create_folds
   ```
2. Launch an Optuna search (example for LightGBM) and immediately train the tuned models with a descriptive alias. Each fold artifact is saved under `models/` and is ready for blending or stacking:
   ```bash
   uv run python -m src.optuna_tuner --model lightgbm --trials 40 --train-best --model-alias lightgbm_optuna
   ```
3. Create a vanilla ensemble submission that averages all folds from the tuned model:
   ```bash
   uv run python -m src.create_submission --models lightgbm_optuna
   ```
4. Or feed several tuned aliases into the stacker to build a meta-model on top of their out-of-fold predictions:
   ```bash
   uv run python -m src.create_stacking_submission --models lightgbm_optuna xgboost_optuna
   ```

5. After you have multiple tuned entries in `results/optuna_best_params.jsonl`, retrain them and build a stacked submission in a single pass (this reuses the exact hyperparameters, trains fold artifacts, logs CV, produces OOF features, and writes the final CSV):
   ```bash
   uv run python -m src.train_tuned_and_stack \
     --models lightgbm_optuna xgboost_optuna catboost_optuna \
     --output submission_tuned_stack.csv
   ```

This workflow stores the best Optuna parameters inside `results/optuna_best_params.jsonl` so you can revisit or reproduce past studies. All training, tuning, and inference scripts now include out-of-fold target encodings (mean repayment rates for key categorical features and their combinations), which often provides a small but consistent boost on top of the engineered numeric interactions.
