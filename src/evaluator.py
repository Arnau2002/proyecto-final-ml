"""Model evaluation utilities.

Fase 3.6 (Baseline evaluation): compute Accuracy and complementary metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union, Mapping

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from src import config


@dataclass(frozen=True)
class EvalResult:
    """Container for evaluation results."""

    accuracy: float
    confusion_matrix: Any
    report: Dict[str, Any]
    text_report: str


def evaluate_classifier(
    model: ClassifierMixin,
    X_test: Union[pd.DataFrame, pd.Series],
    y_test: Union[pd.Series, pd.DataFrame],
    *,
    save_report_path: Optional[Union[str, Path]] = None,
    digits: int = 4,
) -> EvalResult:
    """Evaluate a fitted classifier on a test set.

    Args:
        model: Any fitted sklearn-like classifier.
        X_test: Test features.
        y_test: Test target.
        save_report_path: If set, write the text report to this path.
        digits: Formatting for the text report.

    Returns:
        EvalResult with accuracy, confusion matrix and classification report.
    """

    if isinstance(X_test, pd.Series):
        X_test = X_test.to_frame()
    if isinstance(y_test, pd.DataFrame):
        if y_test.shape[1] != 1:
            raise ValueError("y_test must be a Series or a single-column DataFrame")
        y_test = y_test.iloc[:, 0]

    print("Evaluando modelo...")
    y_pred = model.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    rep_dict = classification_report(y_test, y_pred, output_dict=True, digits=digits)
    rep_text = classification_report(y_test, y_pred, digits=digits)

    print(f"Accuracy: {acc:.{digits}f}")

    if save_report_path is None:
        save_report_path = config.OUTPUTS_DIR / "baseline_report.txt"

    if save_report_path is not None:
        save_report_path = Path(save_report_path)
        save_report_path.parent.mkdir(parents=True, exist_ok=True)
        save_report_path.write_text(
            f"Accuracy: {acc:.{digits}f}\n\nConfusion matrix:\n{cm}\n\nClassification report:\n{rep_text}\n",
            encoding="utf-8",
        )
        print(f"Reporte guardado en: {save_report_path}")

    return EvalResult(accuracy=acc, confusion_matrix=cm, report=rep_dict, text_report=rep_text)


def evaluate_many(
    models: Mapping[str, Any],
    X_test: Union[pd.DataFrame, pd.Series],
    y_test: Union[pd.Series, pd.DataFrame],
    *,
    save_report_path: Optional[Union[str, Path]] = None,
    digits: int = 4,
) -> pd.DataFrame:
    """Evaluate several models and return a comparison table.

    For sklearn models, uses `predict`. For Keras models trained via
    `train_neural_network`, pass the dict return (with key `model`).
    """

    if isinstance(X_test, pd.Series):
        X_test = X_test.to_frame()
    if isinstance(y_test, pd.DataFrame):
        if y_test.shape[1] != 1:
            raise ValueError("y_test must be a Series or a single-column DataFrame")
        y_test = y_test.iloc[:, 0]

    rows = []
    details = []
    for name, model in models.items():
        if isinstance(model, dict) and "model" in model:
            # Keras path
            try:
                import numpy as np

                X_np = X_test.to_numpy(dtype=np.float32)
                y_true = y_test.to_numpy()
                probs = model["model"].predict(X_np, verbose=0)
                if probs.ndim == 2 and probs.shape[1] > 1:
                    y_pred = probs.argmax(axis=1)
                else:
                    y_pred = (probs.reshape(-1) >= 0.5).astype(int)
            except Exception as e:
                raise RuntimeError(f"No se pudo evaluar el modelo Keras '{name}': {e}")
        else:
            y_pred = model.predict(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        rep_text = classification_report(y_test, y_pred, digits=digits)
        rows.append({"model": name, "accuracy": acc})
        details.append((name, acc, cm, rep_text))

    df = pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)

    if save_report_path is None:
        save_report_path = config.OUTPUTS_DIR / "model_comparison.txt"

    if save_report_path is not None:
        save_report_path = Path(save_report_path)
        save_report_path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["MODEL COMPARISON\n", df.to_string(index=False), "\n\n"]
        for name, acc, cm, rep_text in details:
            lines.append(f"=== {name} ===\n")
            lines.append(f"Accuracy: {acc:.{digits}f}\n")
            lines.append(f"Confusion matrix:\n{cm}\n")
            lines.append(f"Classification report:\n{rep_text}\n\n")
        save_report_path.write_text("".join(lines), encoding="utf-8")

    return df
