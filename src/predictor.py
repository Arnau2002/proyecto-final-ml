"""Inference helpers.

This module centralizes artifact loading so inference clients can stay thin.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping, Union

import joblib
import pandas as pd
from sklearn.base import ClassifierMixin

from src import config


def load(path: Union[str, Path]) -> ClassifierMixin:
    """Load a persisted sklearn model (joblib)."""

    return joblib.load(Path(path))


@lru_cache(maxsize=1)
def load_feature_defaults(path: Union[str, Path, None] = None) -> Dict[str, Any]:
    """Load the JSON defaults produced during training."""

    target_path = Path(path) if path is not None else config.FEATURE_DEFAULTS_PATH
    if not target_path.exists():
        raise FileNotFoundError(
            f"No se encontraron los valores por defecto en: {target_path}. Ejecuta src.train_pipeline primero."
        )
    return json.loads(target_path.read_text(encoding="utf-8"))


def build_feature_frame(overrides: Mapping[str, Any], defaults: Dict[str, Any] | None = None) -> pd.DataFrame:
    """Create a single-row DataFrame matching the training schema."""

    defaults = defaults or load_feature_defaults()
    feature_vector = defaults.copy()
    for key, value in overrides.items():
        if key not in feature_vector:
            # Ignore unknown keys to keep robustness inside the UI layer
            continue
        feature_vector[key] = value
    return pd.DataFrame([feature_vector])


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
