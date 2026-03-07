from __future__ import annotations

import time
from pathlib import Path
from typing import Dict

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

from pipeline import (
    HAS_LGBM,
    LINEAR_MODELS,
    TREE_MODELS,
    TARGET_COL,
    build_pipeline,
    choose_best_model,
    cross_validate_model,
    evaluate_regression,
    get_feature_names,
    get_model_zoo,
    load_data,
    save_json,
    split_features_target,
)

DATA_PATH = Path(__file__).resolve().parent.parent / "AB_NYC_2019.csv"
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_DIR = ARTIFACT_DIR / "models"
PLOT_DIR = ARTIFACT_DIR / "plots"
SHAP_DIR = ARTIFACT_DIR / "shap"
CV_SPLITS = 3
SHAP_SAMPLE_SIZE = 500


def ensure_dirs() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    SHAP_DIR.mkdir(parents=True, exist_ok=True)


def to_dense(matrix):
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return matrix


def build_explainer(model_name: str, model, X_background: np.ndarray):
    if model_name in TREE_MODELS:
        return shap.TreeExplainer(model)
    if model_name in LINEAR_MODELS:
        background = shap.sample(X_background, min(250, len(X_background)), random_state=42)
        return shap.LinearExplainer(model, background)
    masker = shap.maskers.Independent(X_background)
    return shap.Explainer(model.predict, masker)


def compute_shap_artifacts(
    model_name: str,
    pipeline,
    X_source: pd.DataFrame,
    feature_names: list[str],
) -> Dict[str, str]:
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocess"]

    X_shap = X_source.sample(n=min(SHAP_SAMPLE_SIZE, len(X_source)), random_state=42)
    X_transformed = to_dense(preprocessor.transform(X_shap))

    explainer = build_explainer(model_name, model, X_transformed)

    if model_name in TREE_MODELS or model_name in LINEAR_MODELS:
        shap_values = explainer.shap_values(X_transformed)
        base_values = explainer.expected_value
    else:
        explanation = explainer(X_transformed)
        shap_values = explanation.values
        base_values = explanation.base_values

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    if np.ndim(base_values) == 0:
        base_values = np.repeat(float(base_values), len(X_transformed))
    elif isinstance(base_values, list):
        base_values = np.array(base_values[0])

    shap_frame = pd.DataFrame(X_transformed, columns=feature_names)
    model_shap_dir = SHAP_DIR / model_name
    model_shap_dir.mkdir(parents=True, exist_ok=True)

    np.save(model_shap_dir / "values.npy", shap_values)
    np.save(model_shap_dir / "base_values.npy", np.asarray(base_values))
    shap_frame.to_parquet(model_shap_dir / "X_transformed.parquet", index=False)

    # Global SHAP importance table
    abs_mean = np.abs(shap_values).mean(axis=0)
    importance = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": abs_mean})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(model_shap_dir / "importance.csv", index=False)

    plt.figure(figsize=(12, 7))
    shap.summary_plot(shap_values, shap_frame, max_display=25, show=False)
    plt.title(f"SHAP Summary (Beeswarm) - {model_name}")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"shap_summary_{model_name}.png", dpi=170)
    plt.close()

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, shap_frame, plot_type="bar", max_display=20, show=False)
    plt.title(f"SHAP Global Importance - {model_name}")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"shap_bar_{model_name}.png", dpi=170)
    plt.close()

    return {
        "shap_values": str(model_shap_dir / "values.npy"),
        "base_values": str(model_shap_dir / "base_values.npy"),
        "transformed_data": str(model_shap_dir / "X_transformed.parquet"),
        "importance": str(model_shap_dir / "importance.csv"),
        "summary_plot": str(PLOT_DIR / f"shap_summary_{model_name}.png"),
        "bar_plot": str(PLOT_DIR / f"shap_bar_{model_name}.png"),
    }


def main() -> None:
    ensure_dirs()

    raw_df = load_data(DATA_PATH)
    dataset = split_features_target(raw_df)

    X_train, X_test, y_train, y_test = train_test_split(
        dataset.X,
        dataset.y,
        test_size=0.2,
        random_state=42,
    )

    models = get_model_zoo(include_lightgbm=True)

    metrics_payload = {}
    shap_index = {}

    for model_name, model in models.items():
        started = time.time()

        pipeline = build_pipeline(model, X_train)
        cv_metrics = cross_validate_model(pipeline, X_train, y_train, n_splits=CV_SPLITS)

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        holdout_metrics = evaluate_regression(y_test, y_pred)

        preprocess = pipeline.named_steps["preprocess"]
        feature_names = get_feature_names(preprocess)

        train_seconds = round(time.time() - started, 2)
        metrics_payload[model_name] = {
            "cv": cv_metrics,
            "holdout": holdout_metrics,
            "feature_count_encoded": int(len(feature_names)),
            "train_seconds": train_seconds,
        }

        joblib.dump(pipeline, MODEL_DIR / f"{model_name}.joblib")

        shap_index[model_name] = compute_shap_artifacts(
            model_name=model_name,
            pipeline=pipeline,
            X_source=X_test,
            feature_names=feature_names,
        )

    best_model = choose_best_model(metrics_payload)

    # Save compact residual diagnostics for charting in the app.
    diagnostics = {}
    for model_name in metrics_payload:
        fitted = joblib.load(MODEL_DIR / f"{model_name}.joblib")
        pred = fitted.predict(X_test)
        diagnostics[model_name] = {
            "y_true": [float(v) for v in y_test.values[:3000]],
            "y_pred": [float(v) for v in pred[:3000]],
            "residuals": [float(v) for v in (y_test.values[:3000] - pred[:3000])],
        }

    summary = {
        "target": TARGET_COL,
        "input_rows": int(len(raw_df)),
        "input_features": dataset.feature_cols,
        "best_model": best_model,
        "models_trained": list(models.keys()),
        "lightgbm_available": bool(HAS_LGBM),
    }

    save_json(ARTIFACT_DIR / "metrics.json", metrics_payload)
    save_json(ARTIFACT_DIR / "summary.json", summary)
    save_json(ARTIFACT_DIR / "shap_index.json", shap_index)
    save_json(ARTIFACT_DIR / "diagnostics.json", diagnostics)


if __name__ == "__main__":
    main()
