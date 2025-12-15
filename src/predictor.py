"""Inference helpers.

This module is intentionally thin: it loads a persisted sklearn model and
applies it to prepared feature matrices.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import joblib
import pandas as pd
from sklearn.base import ClassifierMixin


def load(path: Union[str, Path]) -> ClassifierMixin:
    """Load a persisted sklearn model (joblib)."""

    return joblib.load(Path(path))


def predict(model: ClassifierMixin, X: Union[pd.DataFrame, pd.Series]):
    """Predict labels for input features."""

    if isinstance(X, pd.Series):
        X = X.to_frame()
    return model.predict(X)


def predict_proba(model: ClassifierMixin, X: Union[pd.DataFrame, pd.Series]):
    """Predict class probabilities when the model supports it."""

    if not hasattr(model, "predict_proba"):
        raise AttributeError("Este modelo no soporta predict_proba")
    if isinstance(X, pd.Series):
        X = X.to_frame()
    return model.predict_proba(X)
