"""End-to-end training entrypoint for the LightGBM pipeline."""

from __future__ import annotations

import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

from src import config, data_loader, evaluator, preprocessing


def train_lightgbm_pipeline() -> None:
    df = data_loader.load_raw_data()
    df = data_loader.clean_data(df)

    target_col = "is_canceled"
    if target_col not in df.columns:
        raise ValueError("La columna objetivo 'is_canceled' no existe en el dataset limpiado.")

    y = df[target_col]
    X = df.drop(columns=[target_col])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=config.RANDOM_SEED,
        stratify=y,
    )

    preprocessor = preprocessing.build_preprocessor(X_train)

    classifier = LGBMClassifier(
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=0.0,
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", classifier),
        ]
    )

    print("Entrenando pipeline completo (preprocesamiento + LightGBM)...")
    pipeline.fit(X_train, y_train)
    print("Entrenamiento finalizado.")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, config.BEST_MODEL_PATH)
    print(f"Pipeline guardado en: {config.BEST_MODEL_PATH}")

    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = config.OUTPUTS_DIR / "lightgbm_report.txt"
    evaluator.evaluate_classifier(pipeline, X_test, y_test, save_report_path=report_path)

    defaults = preprocessing.compute_feature_defaults(X)
    preprocessing.save_feature_defaults(defaults, config.FEATURE_DEFAULTS_PATH)
    print(f"Valores por defecto guardados en: {config.FEATURE_DEFAULTS_PATH}")


if __name__ == "__main__":
    train_lightgbm_pipeline()
