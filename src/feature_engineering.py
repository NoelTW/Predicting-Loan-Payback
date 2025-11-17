"""Feature engineering utilities."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from .config import TARGET_COL


def create_features(
    df: DataFrame, *, is_train: bool = True
) -> Tuple[DataFrame, Series | None]:
    """Return engineered features and the target (if requested)."""
    df = df.copy()

    # === NUMERIC FEATURES ===
    df["loan_to_income_ratio"] = df["loan_amount"] / (df["annual_income"] + 1)
    df["monthly_payment_estimate"] = (
        df["loan_amount"] * df["interest_rate"] / 100
    ) / 12
    df["payment_to_income_ratio"] = df["monthly_payment_estimate"] / (
        (df["annual_income"] / 12) + 1
    )
    df["total_debt_estimate"] = df["annual_income"] * df["debt_to_income_ratio"]

    df["debt_interest_product"] = df["debt_to_income_ratio"] * df["interest_rate"]
    df["high_risk_flag"] = (
        (df["debt_to_income_ratio"] > 0.4) & (df["interest_rate"] > 15)
    ).astype(int)
    df["credit_quality_score"] = df["credit_score"] / (df["interest_rate"] + 1)

    df["annual_income_log"] = np.log1p(df["annual_income"])
    df["loan_amount_log"] = np.log1p(df["loan_amount"])

    df["credit_score_squared"] = df["credit_score"] ** 2
    df["debt_to_income_squared"] = df["debt_to_income_ratio"] ** 2

    df["credit_category"] = pd.cut(
        df["credit_score"],
        bins=[0, 650, 700, 750, 850],
        labels=[1, 2, 3, 4],
    ).astype(int)
    df["interest_category"] = pd.cut(
        df["interest_rate"],
        bins=[0, 10, 13, 16, 30],
        labels=[1, 2, 3, 4],
    ).astype(int)

    # === CATEGORICAL FEATURES ===
    education_order = {
        "Other": 0,
        "High School": 1,
        "Bachelor's": 2,
        "Master's": 3,
        "PhD": 4,
    }
    df["education_ordinal"] = df["education_level"].map(education_order)

    grade_order = {
        "A1": 1,
        "A2": 2,
        "A3": 3,
        "A4": 4,
        "A5": 5,
        "B1": 6,
        "B2": 7,
        "B3": 8,
        "B4": 9,
        "B5": 10,
        "C1": 11,
        "C2": 12,
        "C3": 13,
        "C4": 14,
        "C5": 15,
        "D1": 16,
        "D2": 17,
        "D3": 18,
        "D4": 19,
        "D5": 20,
        "E1": 21,
        "E2": 22,
        "E3": 23,
        "E4": 24,
        "E5": 25,
        "F1": 26,
        "F2": 27,
        "F3": 28,
        "F4": 29,
        "F5": 30,
    }
    df["grade_subgrade_ordinal"] = df["grade_subgrade"].map(grade_order)

    df["grade_letter"] = df["grade_subgrade"].str[0]
    df["grade_number"] = df["grade_subgrade"].str[1].astype(int)
    grade_letter_order = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6}
    df["grade_letter_ordinal"] = df["grade_letter"].map(grade_letter_order)
    df["high_grade_risk"] = (df["grade_letter_ordinal"] >= 4).astype(int)

    df = pd.get_dummies(
        df, columns=["gender"], prefix="gender", drop_first=True, dtype=int
    )
    df = pd.get_dummies(
        df, columns=["marital_status"], prefix="marital", drop_first=True, dtype=int
    )
    df = pd.get_dummies(
        df,
        columns=["employment_status"],
        prefix="employment",
        drop_first=True,
        dtype=int,
    )
    df = pd.get_dummies(
        df, columns=["loan_purpose"], prefix="purpose", drop_first=True, dtype=int
    )

    # === INTERACTION FEATURES ===
    df["grade_income_interaction"] = (
        df["grade_subgrade_ordinal"] * df["annual_income_log"]
    )
    df["credit_grade_interaction"] = df["credit_score"] * df["grade_letter_ordinal"]

    # === CLEANUP ===
    drop_cols = ["education_level", "grade_subgrade", "grade_letter", "id"]

    if is_train and TARGET_COL in df.columns:
        y = df[TARGET_COL]
        drop_cols.append(TARGET_COL)
    else:
        y = None

    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    return df, y


__all__ = ["create_features"]
