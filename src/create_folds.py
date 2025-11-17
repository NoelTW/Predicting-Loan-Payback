import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import StratifiedGroupKFold

from .config import (
    ID_COL,
    NUM_FOLDS,
    RANDOM_STATE,
    TARGET_COL,
    TRAIN_FOLDS_FILE,
    TRAINING_FILE,
)


def create_fold_data(df: DataFrame) -> DataFrame:
    """Add a stratified group ``kfold`` column to ``df``."""
    working_df = df.copy()
    working_df["kfold"] = -1

    sgkf = StratifiedGroupKFold(
        n_splits=NUM_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    for fold, (_, valid_idx) in enumerate(
        sgkf.split(working_df, working_df[TARGET_COL], groups=working_df[ID_COL])
    ):
        working_df.loc[valid_idx, "kfold"] = fold

    return working_df


def main():
    if not TRAIN_FOLDS_FILE.is_file():
        print("train_folds.csv file not found. Creating new folds...")
        df = pd.read_csv(TRAINING_FILE)
        df = create_fold_data(df)
        df.to_csv(TRAIN_FOLDS_FILE, index=False)
    else:
        print("train_folds.csv already exists. Skipping fold creation.")


if __name__ == "__main__":
    main()
