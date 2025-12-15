"""Training utilities for ML models.

Roadmap coverage:
  - Fase 3.5: Baseline (Logistic Regression)
  - Fase 4.1: Árboles (Decision Tree, Random Forest)
  - Fase 4.2: Boosting (XGBoost, LightGBM)
  - Fase 4.3: Red Neuronal (Keras/TensorFlow)
  - Fase 4.4: Ajuste de hiperparámetros (helpers for Grid/Random Search)

Designed to be imported from notebooks.
When `save_path` is provided (or omitted), models are persisted with joblib.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src import config


@dataclass(frozen=True)
class TrainResult:
    """Generic return object for training."""

    model: Any
    save_path: Optional[Path]
    best_params: Optional[Dict[str, Any]] = None
    cv_best_score: Optional[float] = None


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _coerce_xy(
    X: Union[pd.DataFrame, pd.Series, np.ndarray],
    y: Union[pd.Series, pd.DataFrame, np.ndarray],
) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]]:
    if isinstance(X, pd.Series):
        X = X.to_frame()
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError("y must be a Series or a single-column DataFrame")
        y = y.iloc[:, 0]
    return X, y


def save_model(model: Any, save_path: Union[str, Path]) -> Path:
    path = Path(save_path)
    _ensure_parent_dir(path)
    joblib.dump(model, path)
    return path


def load_model(path: Union[str, Path]) -> Any:
    """Load a persisted model (joblib)."""

    return joblib.load(Path(path))


# -----------------------------
# Fase 3.5: Baseline
# -----------------------------


def train_logistic_regression(
    X_train: Union[pd.DataFrame, pd.Series, np.ndarray],
    y_train: Union[pd.Series, pd.DataFrame, np.ndarray],
    *,
    max_iter: int = 1000,
    solver: str = "lbfgs",
    class_weight: Optional[str] = None,
    random_state: int = config.RANDOM_SEED,
    save_path: Optional[Union[str, Path]] = None,
) -> TrainResult:
    """Train a Logistic Regression baseline classifier."""

    X_train, y_train = _coerce_xy(X_train, y_train)

    print("Entrenando baseline (Regresión Logística)...")
    model = LogisticRegression(
        max_iter=max_iter,
        solver=solver,
        class_weight=class_weight,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    print("Entrenamiento completado.")

    final_path = Path(save_path) if save_path is not None else (config.MODELS_DIR / "baseline_logreg.joblib")
    save_model(model, final_path)
    print(f"Modelo guardado en: {final_path}")
    return TrainResult(model=model, save_path=final_path)


# -----------------------------
# Fase 4.1: Árboles
# -----------------------------


def train_decision_tree(
    X_train: Union[pd.DataFrame, pd.Series, np.ndarray],
    y_train: Union[pd.Series, pd.DataFrame, np.ndarray],
    *,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    class_weight: Optional[str] = None,
    random_state: int = config.RANDOM_SEED,
    save_path: Optional[Union[str, Path]] = None,
) -> TrainResult:
    X_train, y_train = _coerce_xy(X_train, y_train)
    print("Entrenando Decision Tree...")
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    print("Entrenamiento completado.")

    final_path = Path(save_path) if save_path is not None else (config.MODELS_DIR / "tree_decision.joblib")
    save_model(model, final_path)
    print(f"Modelo guardado en: {final_path}")
    return TrainResult(model=model, save_path=final_path)


def train_random_forest(
    X_train: Union[pd.DataFrame, pd.Series, np.ndarray],
    y_train: Union[pd.Series, pd.DataFrame, np.ndarray],
    *,
    n_estimators: int = 300,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    class_weight: Optional[str] = None,
    n_jobs: int = -1,
    random_state: int = config.RANDOM_SEED,
    save_path: Optional[Union[str, Path]] = None,
) -> TrainResult:
    X_train, y_train = _coerce_xy(X_train, y_train)
    print("Entrenando Random Forest...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    print("Entrenamiento completado.")

    final_path = Path(save_path) if save_path is not None else (config.MODELS_DIR / "tree_random_forest.joblib")
    save_model(model, final_path)
    print(f"Modelo guardado en: {final_path}")
    return TrainResult(model=model, save_path=final_path)


# -----------------------------
# Fase 4.2: Boosting
# -----------------------------


def train_xgboost(
    X_train: Union[pd.DataFrame, pd.Series, np.ndarray],
    y_train: Union[pd.Series, pd.DataFrame, np.ndarray],
    *,
    n_estimators: int = 600,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    subsample: float = 0.9,
    colsample_bytree: float = 0.9,
    reg_lambda: float = 1.0,
    random_state: int = config.RANDOM_SEED,
    save_path: Optional[Union[str, Path]] = None,
) -> TrainResult:
    """Train an XGBoost classifier.

    Notes:
        - Requires `xgboost` in requirements.txt.
    """

    from xgboost import XGBClassifier  # lazy import

    X_train, y_train = _coerce_xy(X_train, y_train)
    print("Entrenando XGBoost...")
    model = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        random_state=random_state,
        eval_metric="logloss",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print("Entrenamiento completado.")

    final_path = Path(save_path) if save_path is not None else (config.MODELS_DIR / "boost_xgboost.joblib")
    save_model(model, final_path)
    print(f"Modelo guardado en: {final_path}")
    return TrainResult(model=model, save_path=final_path)


def train_lightgbm(
    X_train: Union[pd.DataFrame, pd.Series, np.ndarray],
    y_train: Union[pd.Series, pd.DataFrame, np.ndarray],
    *,
    n_estimators: int = 1200,
    learning_rate: float = 0.03,
    num_leaves: int = 31,
    max_depth: int = -1,
    subsample: float = 0.9,
    colsample_bytree: float = 0.9,
    reg_lambda: float = 0.0,
    random_state: int = config.RANDOM_SEED,
    save_path: Optional[Union[str, Path]] = None,
) -> TrainResult:
    """Train a LightGBM classifier.

    Notes:
        - Requires `lightgbm` in requirements.txt.
    """

    from lightgbm import LGBMClassifier  # lazy import

    X_train, y_train = _coerce_xy(X_train, y_train)
    print("Entrenando LightGBM...")
    model = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print("Entrenamiento completado.")

    final_path = Path(save_path) if save_path is not None else (config.MODELS_DIR / "boost_lightgbm.joblib")
    save_model(model, final_path)
    print(f"Modelo guardado en: {final_path}")
    return TrainResult(model=model, save_path=final_path)


# -----------------------------
# Fase 4.3: Red Neuronal
# -----------------------------


def train_neural_network(
    X_train: Union[pd.DataFrame, pd.Series, np.ndarray],
    y_train: Union[pd.Series, pd.DataFrame, np.ndarray],
    X_val: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]] = None,
    y_val: Optional[Union[pd.Series, pd.DataFrame, np.ndarray]] = None,
    *,
    epochs: int = 40,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    hidden_units: Tuple[int, int] = (128, 64),
    dropout: float = 0.2,
    save_path: Optional[Union[str, Path]] = None,
) -> TrainResult:
    """Train a simple MLP for tabular classification.

    This keeps the architecture intentionally modest and stable for coursework.
    If your target is not binary, it automatically switches to softmax.
    """

    import tensorflow as tf  # lazy import

    X_train, y_train = _coerce_xy(X_train, y_train)
    if isinstance(X_train, pd.DataFrame):
        X_train_np = X_train.to_numpy(dtype=np.float32)
    else:
        X_train_np = np.asarray(X_train, dtype=np.float32)

    y_train_np = np.asarray(y_train)

    n_classes = int(np.unique(y_train_np).shape[0])
    is_binary = n_classes == 2

    if X_val is not None and y_val is not None:
        X_val, y_val = _coerce_xy(X_val, y_val)
        X_val_np = X_val.to_numpy(dtype=np.float32) if isinstance(X_val, pd.DataFrame) else np.asarray(X_val, dtype=np.float32)
        y_val_np = np.asarray(y_val)
    else:
        X_val_np, y_val_np = None, None

    print("Entrenando red neuronal (MLP)...")

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(X_train_np.shape[1],)),
            tf.keras.layers.Dense(hidden_units[0], activation="relu"),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(hidden_units[1], activation="relu"),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(1 if is_binary else n_classes, activation="sigmoid" if is_binary else "softmax"),
        ]
    )

    loss = "binary_crossentropy" if is_binary else "sparse_categorical_crossentropy"
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
    ]

    history = model.fit(
        X_train_np,
        y_train_np,
        validation_data=(X_val_np, y_val_np) if X_val_np is not None else None,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    print("Entrenamiento completado.")

    # Keras models are not joblib-friendly. Save as .keras by default.
    final_path = Path(save_path) if save_path is not None else (config.MODELS_DIR / "nn_mlp.keras")
    _ensure_parent_dir(final_path)
    model.save(final_path)
    print(f"Modelo guardado en: {final_path}")

    # Return also history for optional plotting.
    return TrainResult(model={"model": model, "history": history.history}, save_path=final_path)


# -----------------------------
# Fase 4.4: Hiperparámetros (helpers)
# -----------------------------


def tune_with_grid_search(
    estimator: BaseEstimator,
    param_grid: Dict[str, Any],
    X_train: Union[pd.DataFrame, pd.Series, np.ndarray],
    y_train: Union[pd.Series, pd.DataFrame, np.ndarray],
    *,
    scoring: str = "accuracy",
    cv: int = 5,
    n_jobs: int = -1,
    save_path: Optional[Union[str, Path]] = None,
) -> TrainResult:
    X_train, y_train = _coerce_xy(X_train, y_train)
    print("GridSearchCV...")
    gs = GridSearchCV(estimator, param_grid=param_grid, scoring=scoring, cv=cv, n_jobs=n_jobs)
    gs.fit(X_train, y_train)
    print(f"Best CV score: {gs.best_score_:.4f}")
    print(f"Best params: {gs.best_params_}")

    best_model = gs.best_estimator_
    final_path = None
    if save_path is not None:
        final_path = Path(save_path)
        save_model(best_model, final_path)
        print(f"Modelo ajustado guardado en: {final_path}")

    return TrainResult(model=best_model, save_path=final_path, best_params=gs.best_params_, cv_best_score=float(gs.best_score_))


def tune_with_random_search(
    estimator: BaseEstimator,
    param_distributions: Dict[str, Any],
    X_train: Union[pd.DataFrame, pd.Series, np.ndarray],
    y_train: Union[pd.Series, pd.DataFrame, np.ndarray],
    *,
    scoring: str = "accuracy",
    cv: int = 5,
    n_iter: int = 30,
    n_jobs: int = -1,
    random_state: int = config.RANDOM_SEED,
    save_path: Optional[Union[str, Path]] = None,
) -> TrainResult:
    X_train, y_train = _coerce_xy(X_train, y_train)
    print("RandomizedSearchCV...")
    rs = RandomizedSearchCV(
        estimator,
        param_distributions=param_distributions,
        scoring=scoring,
        cv=cv,
        n_iter=n_iter,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    rs.fit(X_train, y_train)
    print(f"Best CV score: {rs.best_score_:.4f}")
    print(f"Best params: {rs.best_params_}")

    best_model = rs.best_estimator_
    final_path = None
    if save_path is not None:
        final_path = Path(save_path)
        save_model(best_model, final_path)
        print(f"Modelo ajustado guardado en: {final_path}")

    return TrainResult(model=best_model, save_path=final_path, best_params=rs.best_params_, cv_best_score=float(rs.best_score_))
