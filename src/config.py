"""Project level configuration."""

from pathlib import Path

# NOTE: config.py lives in src/, so parent.parent points at the project root.
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "input"
RAW_DATA_DIR = INPUT_DIR / "raw_data"
PROCESSED_DATA_DIR = INPUT_DIR / "processed"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

TRAINING_FILE = RAW_DATA_DIR / "train.csv"
TEST_FILE = RAW_DATA_DIR / "test.csv"
TRAIN_FOLDS_FILE = PROCESSED_DATA_DIR / "train_folds.csv"

ID_COL = "id"
TARGET_COL = "loan_paid_back"

NUM_FOLDS = 5
RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "annual_income",
    "debt_to_income_ratio",
    "credit_score",
    "loan_amount",
    "interest_rate",
]

CATEGORICAL_FEATURES = [
    "gender",
    "marital_status",
    "education_level",
    "employment_status",
    "loan_purpose",
    "grade_subgrade",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

for path in (
    INPUT_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    RESULTS_DIR,
):
    path.mkdir(parents=True, exist_ok=True)
