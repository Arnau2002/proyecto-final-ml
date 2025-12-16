"""Preprocessing utilities for the ML pipeline.

This module centralizes feature engineering pieces that must be
shared between training notebooks/scripts and inference code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def infer_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Derive numeric and categorical feature lists from a DataFrame."""

    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = [col for col in X.columns if col not in categorical]
    return numeric, categorical


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create a ColumnTransformer that handles scaling + encoding."""

    numeric_cols, categorical_cols = infer_feature_types(X)

    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_cols,
            )
        )

    return ColumnTransformer(transformers=transformers)


def compute_feature_defaults(X: pd.DataFrame) -> Dict[str, object]:
    """Return median/mode defaults for every raw feature."""

    defaults: Dict[str, object] = {}
    for column in X.columns:
        series = X[column]
        if pd.api.types.is_numeric_dtype(series):
            defaults[column] = float(series.median())
        else:
            mode_series = series.mode(dropna=True)
            defaults[column] = mode_series.iloc[0] if not mode_series.empty else ""
    return defaults


def save_feature_defaults(defaults: Dict[str, object], destination: Path) -> None:
    """Persist the defaults dictionary as JSON."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(defaults, indent=2), encoding="utf-8")
