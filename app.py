from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from pipeline import HAS_LGBM, build_pipeline, cross_validate_model, get_model_zoo, load_data, split_features_target
import eda

DATA_PATH = Path(__file__).resolve().parent.parent / "AB_NYC_2019.csv"
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_DIR = ARTIFACT_DIR / "models"
PLOT_DIR = ARTIFACT_DIR / "plots"

st.set_page_config(page_title="NYC Airbnb Model Studio", layout="wide")

# === Assignment-configurable variables ===
DATASET_DESCRIPTION = (
    "NYC Airbnb listings dataset with property, location, and host attributes."
)
TARGET_COLUMN = "price"
TARGET_TYPE_OVERRIDE = None  # Options: None, "classification", "regression"
TASK_TYPE = "auto"  # Options: "auto", "classification", "regression"
DATASET_SOURCE_DESCRIPTION = (
    "NYC Airbnb Open Data (AB_NYC_2019) compiled from Inside Airbnb listings."
)
PREDICTION_TASK_DESCRIPTION = (
    "Predict nightly listing price to support pricing strategy and market analysis."
)
WHY_IT_MATTERS = (
    "Accurate price prediction helps hosts and platforms set competitive rates and improve market efficiency."
)
APPROACH_SUMMARY = (
    "We profile the data, compare multiple models using a held-out test set, and explain the best tree-based model with SHAP."
)
KEY_FINDINGS_SUMMARY = (
    "Tree-based ensemble models deliver the most accurate predictions while highlighting key drivers such as location and room type."
)


@st.cache_data
def load_dataset() -> pd.DataFrame:
    return load_data(DATA_PATH)


@st.cache_data
def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_shap_payload(model_name: str):
    shap_dir = ARTIFACT_DIR / "shap" / model_name
    values_path = shap_dir / "values.npy"
    data_path = shap_dir / "X_transformed.parquet"
    importance_path = shap_dir / "importance.csv"

    if not values_path.exists() or not data_path.exists() or not importance_path.exists():
        return None, None, None

    values = np.load(values_path, allow_pickle=True)
    data = pd.read_parquet(data_path)
    importance = pd.read_csv(importance_path)
    return values, data, importance


@st.cache_resource
def load_trained_model(model_name: str):
    model_path = MODEL_DIR / f"{model_name}.joblib"
    if model_path.exists():
        return joblib.load(model_path)
    return None


@st.cache_data
def run_feature_playground(model_name: str, selected_features: list[str], n_splits: int):
    df = load_dataset()
    ds = split_features_target(df)
    X = ds.X[selected_features]
    y = ds.y

    model = get_model_zoo(include_lightgbm=HAS_LGBM)[model_name]
    pipeline = build_pipeline(model, X)
    return cross_validate_model(pipeline, X, y, n_splits=n_splits)


def classify_columns(df: pd.DataFrame, target_col: str):
    num_cols = []
    cat_cols = []
    bool_cols = []
    dt_cols = []
    id_like_cols = []

    for col in df.columns:
        if col == target_col:
            continue
        series = df[col]
        if pd.api.types.is_bool_dtype(series):
            bool_cols.append(col)
            continue
        if pd.api.types.is_datetime64_any_dtype(series):
            dt_cols.append(col)
            continue
        if pd.api.types.is_numeric_dtype(series):
            num_cols.append(col)
        else:
            cat_cols.append(col)

    total_rows = max(len(df), 1)
    for col in df.columns:
        if col == target_col:
            continue
        series = df[col]
        unique_ratio = series.nunique(dropna=True) / total_rows
        if col.lower().endswith("id") or "id_" in col.lower() or unique_ratio > 0.9:
            id_like_cols.append(col)

    num_cols = [c for c in num_cols if c not in id_like_cols]
    cat_cols = [c for c in cat_cols if c not in id_like_cols]

    return {
        "numeric": num_cols,
        "categorical": cat_cols,
        "boolean": bool_cols,
        "datetime": dt_cols,
        "id_like": id_like_cols,
    }


def determine_target_type(df: pd.DataFrame, target_col: str, override: str | None):
    if override in {"classification", "regression"}:
        return override
    if target_col not in df.columns:
        return "regression"

    target = df[target_col]
    unique_vals = target.nunique(dropna=True)

    if not pd.api.types.is_numeric_dtype(target):
        return "classification"
    if unique_vals <= 15:
        return "classification"
    return "regression"


def detect_task_type(y: pd.Series, override: str | None = None) -> str:
    if override in {"classification", "regression"}:
        return override
    if not pd.api.types.is_numeric_dtype(y):
        return "classification"
    unique_vals = y.nunique(dropna=True)
    if unique_vals <= 15:
        return "classification"
    return "regression"


def get_column_types(df: pd.DataFrame, target_col: str) -> dict:
    numeric_cols = []
    categorical_cols = []
    for col in df.columns:
        if col == target_col:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    return {"numeric": numeric_cols, "categorical": categorical_cols}


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str], dense_output: bool = False):
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    try:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=not dense_output)
    except TypeError:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse=not dense_output)

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", onehot),
        ]
    )

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_transformer, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_transformer, categorical_cols))

    if not transformers:
        return None

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0 if dense_output else 0.3,
    )


@st.cache_data
def prepare_data(df: pd.DataFrame, target_col: str, task_type: str):
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()
    stratify = None
    if task_type == "classification" and y.nunique(dropna=True) > 1:
        if y.value_counts().min() >= 2:
            stratify = y
    return train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=stratify,
    )


def evaluate_classification(y_true, y_pred, y_score=None):
    y_true_series = pd.Series(y_true)
    average = "binary" if y_true_series.nunique() == 2 else "weighted"
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
        "roc_auc": None,
    }
    if y_score is not None:
        try:
            if y_true_series.nunique() == 2:
                if not pd.api.types.is_numeric_dtype(y_true_series):
                    y_true_enc = y_true_series.astype("category").cat.codes
                else:
                    y_true_enc = y_true_series
                metrics["roc_auc"] = roc_auc_score(y_true_enc, y_score)
            else:
                y_true_enc = y_true_series.astype("category").cat.codes
                metrics["roc_auc"] = roc_auc_score(y_true_enc, y_score, multi_class="ovr", average="weighted")
        except Exception:
            metrics["roc_auc"] = None
    return metrics


def evaluate_regression(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mse),
        "r2": r2_score(y_true, y_pred),
    }


def plot_confusion_matrix(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)


def plot_roc_curve(y_true, y_score):
    fig, ax = plt.subplots(figsize=(5, 4))
    try:
        if hasattr(y_score, "ndim") and y_score.ndim > 1:
            st.info("ROC curve visualization is shown for binary classification only.")
            return
        RocCurveDisplay.from_predictions(y_true, y_score, ax=ax)
        ax.set_title("ROC Curve")
        st.pyplot(fig)
    except Exception:
        st.info("ROC curve not available for this model or target configuration.")


def plot_predicted_vs_actual(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_true, y_pred, alpha=0.6)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs Actual")
    st.pyplot(fig)


def get_cv_strategy(task_type: str, y_train: pd.Series):
    if task_type == "classification":
        class_counts = y_train.value_counts()
        min_class = class_counts.min() if not class_counts.empty else 0
        n_splits = min(5, min_class)
        if n_splits < 2:
            return None
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    n_splits = min(5, len(y_train))
    if n_splits < 2:
        return None
    return KFold(n_splits=n_splits, shuffle=True, random_state=42)


def train_baseline_model(task_type, preprocessor, X_train, y_train, X_test, y_test):
    if task_type == "classification":
        model = LogisticRegression(max_iter=2000, random_state=42, solver="saga")
        model_name = "Logistic Regression (Baseline)"
    else:
        model = LinearRegression()
        model_name = "Linear Regression (Baseline)"

    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_score = None
    if task_type == "classification":
        if hasattr(pipeline, "predict_proba"):
            y_score = pipeline.predict_proba(X_test)
            if y_score.ndim == 2 and y_score.shape[1] == 2:
                y_score = y_score[:, 1]
        elif hasattr(pipeline, "decision_function"):
            y_score = pipeline.decision_function(X_test)
        metrics = evaluate_classification(y_test, y_pred, y_score)
    else:
        metrics = evaluate_regression(y_test, y_pred)

    return {
        "model_name": model_name,
        "model": pipeline,
        "metrics": metrics,
        "y_pred": y_pred,
        "y_score": y_score,
    }


def train_decision_tree_model(task_type, preprocessor, X_train, y_train, X_test, y_test, cv):
    if task_type == "classification":
        model = DecisionTreeClassifier(random_state=42)
        scoring = "f1" if y_train.nunique() == 2 else "f1_weighted"
    else:
        model = DecisionTreeRegressor(random_state=42)
        scoring = "neg_mean_squared_error"

    param_grid = {
        "model__max_depth": [3, 5, 7, 10],
        "model__min_samples_leaf": [5, 10, 20, 50],
    }

    if cv is None:
        return None

    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    y_score = None
    if task_type == "classification":
        if hasattr(best_model, "predict_proba"):
            y_score = best_model.predict_proba(X_test)
            if y_score.ndim == 2 and y_score.shape[1] == 2:
                y_score = y_score[:, 1]
        elif hasattr(best_model, "decision_function"):
            y_score = best_model.decision_function(X_test)
        metrics = evaluate_classification(y_test, y_pred, y_score)
    else:
        metrics = evaluate_regression(y_test, y_pred)

    return {
        "model_name": "Decision Tree",
        "model": best_model,
        "metrics": metrics,
        "y_pred": y_pred,
        "y_score": y_score,
        "best_params": grid.best_params_,
        "cv_best_score": grid.best_score_,
    }


def train_random_forest_model(task_type, preprocessor, X_train, y_train, X_test, y_test, cv):
    if task_type == "classification":
        model = RandomForestClassifier(random_state=42)
        scoring = "f1" if y_train.nunique() == 2 else "f1_weighted"
    else:
        model = RandomForestRegressor(random_state=42)
        scoring = "neg_mean_squared_error"

    param_grid = {
        "model__n_estimators": [50, 100, 200],
        "model__max_depth": [3, 5, 8],
    }

    if cv is None:
        return None

    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    y_score = None
    if task_type == "classification":
        if hasattr(best_model, "predict_proba"):
            y_score = best_model.predict_proba(X_test)
            if y_score.ndim == 2 and y_score.shape[1] == 2:
                y_score = y_score[:, 1]
        elif hasattr(best_model, "decision_function"):
            y_score = best_model.decision_function(X_test)
        metrics = evaluate_classification(y_test, y_pred, y_score)
    else:
        metrics = evaluate_regression(y_test, y_pred)

    return {
        "model_name": "Random Forest",
        "model": best_model,
        "metrics": metrics,
        "y_pred": y_pred,
        "y_score": y_score,
        "best_params": grid.best_params_,
        "cv_best_score": grid.best_score_,
    }


def train_boosted_model(task_type, preprocessor, preprocessor_dense, X_train, y_train, X_test, y_test, cv):
    model_name = None
    model = None
    param_grid = None
    scoring = "f1" if task_type == "classification" and y_train.nunique() == 2 else "f1_weighted"
    if task_type == "regression":
        scoring = "neg_mean_squared_error"

    try:
        from xgboost import XGBClassifier, XGBRegressor  # type: ignore

        if task_type == "classification":
            model = XGBClassifier(
                random_state=42,
                eval_metric="logloss",
                use_label_encoder=False,
            )
        else:
            model = XGBRegressor(random_state=42)
        model_name = "XGBoost"
        param_grid = {
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [3, 4, 5, 6],
            "model__learning_rate": [0.01, 0.05, 0.1],
        }
    except Exception:
        try:
            from lightgbm import LGBMClassifier, LGBMRegressor  # type: ignore

            if task_type == "classification":
                model = LGBMClassifier(random_state=42)
            else:
                model = LGBMRegressor(random_state=42)
            model_name = "LightGBM"
            param_grid = {
                "model__n_estimators": [50, 100, 200],
                "model__max_depth": [3, 4, 5, 6],
                "model__learning_rate": [0.01, 0.05, 0.1],
            }
        except Exception:
            if task_type == "classification":
                model = HistGradientBoostingClassifier(random_state=42)
            else:
                model = HistGradientBoostingRegressor(random_state=42)
            model_name = "HistGradientBoosting (Fallback)"
            param_grid = {
                "model__max_iter": [50, 100, 200],
                "model__max_depth": [3, 4, 5],
                "model__learning_rate": [0.01, 0.05, 0.1],
            }

    if cv is None:
        return None

    if isinstance(model, (HistGradientBoostingClassifier, HistGradientBoostingRegressor)):
        active_preprocessor = preprocessor_dense
    else:
        active_preprocessor = preprocessor

    pipeline = Pipeline(steps=[("preprocess", active_preprocessor), ("model", model)])
    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    y_score = None
    if task_type == "classification":
        if hasattr(best_model, "predict_proba"):
            y_score = best_model.predict_proba(X_test)
            if y_score.ndim == 2 and y_score.shape[1] == 2:
                y_score = y_score[:, 1]
        elif hasattr(best_model, "decision_function"):
            y_score = best_model.decision_function(X_test)
        metrics = evaluate_classification(y_test, y_pred, y_score)
    else:
        metrics = evaluate_regression(y_test, y_pred)

    return {
        "model_name": model_name,
        "model": best_model,
        "metrics": metrics,
        "y_pred": y_pred,
        "y_score": y_score,
        "best_params": grid.best_params_,
        "cv_best_score": grid.best_score_,
    }


def train_mlp_model(task_type, preprocessor, X_train, y_train, X_test, y_test):
    use_tf = False
    history = None
    model_name = None
    y_score = None

    try:
        import tensorflow as tf  # type: ignore
        from tensorflow import keras  # type: ignore

        use_tf = True
    except Exception:
        use_tf = False

    dense_preprocessor = preprocessor
    if dense_preprocessor is None:
        return None

    if use_tf:
        model_name = "Neural Network (Keras)"
        tf.random.set_seed(42)
        X_train_p = dense_preprocessor.fit_transform(X_train)
        X_test_p = dense_preprocessor.transform(X_test)
        X_train_p = np.asarray(X_train_p)
        X_test_p = np.asarray(X_test_p)

        if task_type == "classification":
            n_classes = pd.Series(y_train).nunique()
            if n_classes == 2:
                y_train_enc = y_train.astype(int) if pd.api.types.is_numeric_dtype(y_train) else pd.Series(y_train).astype("category").cat.codes
                y_test_enc = y_test.astype(int) if pd.api.types.is_numeric_dtype(y_test) else pd.Series(y_test).astype("category").cat.codes
                output_units = 1
                activation = "sigmoid"
                loss_fn = "binary_crossentropy"
            else:
                y_train_enc = pd.Series(y_train).astype("category").cat.codes
                y_test_enc = pd.Series(y_test).astype("category").cat.codes
                output_units = n_classes
                activation = "softmax"
                loss_fn = "sparse_categorical_crossentropy"

            model = keras.Sequential(
                [
                    keras.layers.Input(shape=(X_train_p.shape[1],)),
                    keras.layers.Dense(128, activation="relu"),
                    keras.layers.Dense(128, activation="relu"),
                    keras.layers.Dense(output_units, activation=activation),
                ]
            )
            model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
        else:
            y_train_enc = y_train.values
            y_test_enc = y_test.values
            model = keras.Sequential(
                [
                    keras.layers.Input(shape=(X_train_p.shape[1],)),
                    keras.layers.Dense(128, activation="relu"),
                    keras.layers.Dense(128, activation="relu"),
                    keras.layers.Dense(1, activation="linear"),
                ]
            )
            model.compile(optimizer="adam", loss="mse", metrics=["mse"])

        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        )
        history = model.fit(
            X_train_p,
            y_train_enc,
            validation_split=0.2,
            epochs=50,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0,
        )
        y_pred = model.predict(X_test_p, verbose=0)

        if task_type == "classification":
            if output_units == 1:
                y_score = y_pred.ravel()
                y_pred_labels = (y_score >= 0.5).astype(int)
            else:
                y_score = y_pred
                y_pred_labels = y_score.argmax(axis=1)
            metrics = evaluate_classification(y_test_enc, y_pred_labels, y_score)
        else:
            y_pred = y_pred.ravel()
            metrics = evaluate_regression(y_test_enc, y_pred)

        return {
            "model_name": model_name,
            "model": model,
            "metrics": metrics,
            "y_pred": y_pred_labels if task_type == "classification" else y_pred,
            "y_score": y_score,
            "history": history,
            "used_tf": True,
            "y_true": y_test_enc if task_type == "classification" else y_test_enc,
        }

    model_name = "Neural Network (MLPClassifier/Regressor Fallback)"
    densify = FunctionTransformer(lambda x: x.toarray() if hasattr(x, "toarray") else x)
    if task_type == "classification":
        model = MLPClassifier(random_state=42, max_iter=500)
    else:
        model = MLPRegressor(random_state=42, max_iter=500)

    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("densify", densify), ("model", model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    if task_type == "classification":
        if hasattr(pipeline, "predict_proba"):
            y_score = pipeline.predict_proba(X_test)
            if y_score.ndim == 2 and y_score.shape[1] == 2:
                y_score = y_score[:, 1]
        elif hasattr(pipeline, "decision_function"):
            y_score = pipeline.decision_function(X_test)
        metrics = evaluate_classification(y_test, y_pred, y_score)
    else:
        metrics = evaluate_regression(y_test, y_pred)

    return {
        "model_name": model_name,
        "model": pipeline,
        "metrics": metrics,
        "y_pred": y_pred,
        "y_score": y_score,
        "history": None,
        "used_tf": False,
    }


def build_model_comparison_table(results: list[dict], task_type: str) -> pd.DataFrame:
    rows = []
    for result in results:
        metrics = result["metrics"]
        row = {"model": result["model_name"]}
        if task_type == "classification":
            row.update(
                {
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "roc_auc": metrics.get("roc_auc"),
                }
            )
        else:
            row.update(
                {
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "r2": metrics["r2"],
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def dataset_intro_table(df: pd.DataFrame, col_types: dict, target_col: str):
    rows = [
        {"Metric": "Rows", "Value": f"{len(df):,}"},
        {"Metric": "Columns (total)", "Value": f"{df.shape[1]:,}"},
        {"Metric": "Numerical features", "Value": f"{len(col_types['numeric']):,}"},
        {"Metric": "Categorical features", "Value": f"{len(col_types['categorical']):,}"},
        {"Metric": "Boolean features", "Value": f"{len(col_types['boolean']):,}"},
        {"Metric": "Datetime features", "Value": f"{len(col_types['datetime']):,}"},
        {"Metric": "ID-like columns (excluded)", "Value": f"{len(col_types['id_like']):,}"},
        {"Metric": "Target column", "Value": target_col},
    ]
    return pd.DataFrame(rows)


def summarize_balance(counts: pd.Series):
    if counts.empty:
        return "Target distribution could not be computed."
    ratio = counts.max() / counts.min() if counts.min() > 0 else np.inf
    if ratio <= 1.5:
        return "The classes appear reasonably balanced, so standard modeling should be stable."
    return (
        "The classes are imbalanced. Consider stratified splits and techniques such as class weights "
        "or resampling (e.g., SMOTE) to reduce bias toward majority classes."
    )


def interpret_regression_distribution(series: pd.Series):
    desc = series.describe()
    skew = series.skew()
    comment = "The target shows a "
    if skew > 1:
        comment += "strong right-skew, suggesting a long tail of high values."
    elif skew < -1:
        comment += "strong left-skew, suggesting a long tail of low values."
    else:
        comment += "moderate spread without extreme skew."
    comment += (
        f" The typical value is around {desc['50%']:.2f} with a spread (std) of {desc['std']:.2f}."
    )
    if (series > desc["75%"] + 1.5 * (desc["75%"] - desc["25%"])).sum() > 0:
        comment += " Potential outliers are visible beyond the upper quartile."
    return comment


def top_numeric_features(df: pd.DataFrame, target_col: str, num_cols: list[str], target_type: str, k: int = 5):
    if not num_cols:
        return []
    X = df[num_cols].copy()
    X = X.select_dtypes(include=[np.number]).fillna(X.median(numeric_only=True))
    y = df[target_col]

    if target_type == "classification":
        y_enc = LabelEncoder().fit_transform(y.astype(str))
        scores = mutual_info_classif(X, y_enc, discrete_features=False, random_state=42)
    else:
        scores = mutual_info_regression(X, y, random_state=42)

    score_df = pd.DataFrame({"feature": X.columns, "score": scores})
    score_df = score_df.sort_values("score", ascending=False)
    return score_df["feature"].head(k).tolist()


def top_categorical_features(df: pd.DataFrame, target_col: str, cat_cols: list[str], k: int = 2):
    if not cat_cols:
        return []
    scores = []
    target = df[target_col].astype(str)
    for col in cat_cols:
        table = pd.crosstab(df[col].astype(str), target)
        if table.size == 0:
            continue
        chi2 = ((table - table.mean()) ** 2 / (table.mean() + 1e-9)).sum().sum()
        scores.append((col, chi2))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return [c for c, _ in scores[:k]]


def render_part1_descriptive(df: pd.DataFrame):
    st.header("Part 1: Descriptive Analytics")

    target_col = TARGET_COLUMN if TARGET_COLUMN in df.columns else df.columns[-1]
    col_types = classify_columns(df, target_col)
    target_type = determine_target_type(df, target_col, TARGET_TYPE_OVERRIDE)

    st.subheader("1.1 Dataset Introduction")
    st.markdown(
        f"""
        **Dataset source:** {DATASET_SOURCE_DESCRIPTION}  
        **Prediction task:** {PREDICTION_TASK_DESCRIPTION}  
        **Target / dependent variable:** `{target_col}`
        """
    )
    summary_table = dataset_intro_table(df, col_types, target_col)
    st.table(summary_table)

    if col_types["numeric"]:
        st.caption(f"Numeric features: {', '.join(col_types['numeric'][:12])}" + (" ..." if len(col_types["numeric"]) > 12 else ""))
    if col_types["categorical"]:
        st.caption(f"Categorical features: {', '.join(col_types['categorical'][:12])}" + (" ..." if len(col_types["categorical"]) > 12 else ""))

    st.subheader("1.2 Target Distribution")
    target_series = df[target_col].dropna()
    if target_type == "classification":
        counts = target_series.astype(str).value_counts()
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=counts.index, y=counts.values, ax=ax, palette="viridis")
        ax.set_title("Target Class Frequency")
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=30)
        st.pyplot(fig)

        count_df = pd.DataFrame({"class": counts.index, "count": counts.values})
        count_df["percent"] = (count_df["count"] / count_df["count"].sum() * 100).round(2)
        st.dataframe(count_df, use_container_width=True)
        st.caption(summarize_balance(counts))
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(target_series, bins=30, kde=True, ax=ax, color="#1f77b4")
        ax.set_title("Target Distribution")
        ax.set_xlabel(target_col)
        st.pyplot(fig)
        st.caption(interpret_regression_distribution(target_series))

    st.subheader("1.3 Feature Distributions and Relationships")
    numeric_cols = col_types["numeric"]
    categorical_cols = col_types["categorical"]

    top_nums = top_numeric_features(df, target_col, numeric_cols, target_type, k=5)
    top_cats = top_categorical_features(df, target_col, categorical_cols, k=2)

    plots_rendered = 0

    if top_nums:
        feature = top_nums[0]
        fig, ax = plt.subplots(figsize=(7, 4))
        if target_type == "classification":
            sns.violinplot(x=df[target_col].astype(str), y=df[feature], ax=ax, inner="quartile")
            ax.set_title(f"{feature} by Target Class")
            ax.set_xlabel("Target Class")
        else:
            sns.scatterplot(x=df[feature], y=df[target_col], ax=ax, alpha=0.5)
            ax.set_title(f"{feature} vs {target_col}")
            ax.set_ylabel(target_col)
        st.pyplot(fig)
        st.caption(
            f"The distribution of `{feature}` shows meaningful variation with the target. "
            "This pattern suggests the feature is informative and could contribute predictive signal."
        )
        plots_rendered += 1

    if len(top_nums) > 1:
        feature = top_nums[1]
        fig, ax = plt.subplots(figsize=(7, 4))
        if target_type == "classification":
            sns.boxplot(x=df[target_col].astype(str), y=df[feature], ax=ax)
            ax.set_title(f"{feature} Spread Across Classes")
        else:
            sns.scatterplot(x=df[feature], y=df[target_col], ax=ax, alpha=0.5)
            ax.set_title(f"{feature} vs {target_col}")
        st.pyplot(fig)
        st.caption(
            f"`{feature}` shows distinct central tendencies across the target, "
            "indicating potential differentiation power for modeling."
        )
        plots_rendered += 1

    if len(top_nums) >= 2:
        feature_x, feature_y = top_nums[:2]
        fig = px.scatter(
            df,
            x=feature_x,
            y=feature_y,
            color=df[target_col].astype(str) if target_type == "classification" else None,
            title=f"{feature_x} vs {feature_y}",
            opacity=0.6,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "The joint pattern between these two numeric features highlights clusters and gradients, "
            "which can reveal nonlinear structure useful for model selection."
        )
        plots_rendered += 1

    if top_cats:
        cat = top_cats[0]
        if target_type == "classification":
            cat_counts = (
                df.groupby([cat, target_col]).size().reset_index(name="count")
            )
            fig = px.bar(
                cat_counts,
                x=cat,
                y="count",
                color=target_col,
                barmode="group",
                title=f"{cat} by Target Class",
            )
        else:
            fig = px.box(
                df,
                x=cat,
                y=target_col,
                title=f"{target_col} by {cat}",
            )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"The categorical feature `{cat}` shows systematic differences across the target, "
            "which may signal segmentation effects worth capturing in modeling."
        )
        plots_rendered += 1

    if plots_rendered < 4 and len(numeric_cols) >= 3:
        subset = numeric_cols[:3]
        pair_df = df[subset].copy()
        hue_col = None
        if target_type == "classification":
            pair_df[target_col] = df[target_col].astype(str)
            hue_col = target_col
        else:
            pair_df[target_col] = df[target_col]
        fig = sns.pairplot(pair_df.dropna(), hue=hue_col)
        st.pyplot(fig.fig)
        st.caption(
            "The pairwise scatter matrix highlights linear and nonlinear relationships across numeric features, "
            "providing a compact view of potential interactions."
        )
        plots_rendered += 1

    if plots_rendered < 4 and len(numeric_cols) >= 2:
        feature_x, feature_y = numeric_cols[:2]
        fig = px.scatter(
            df,
            x=feature_x,
            y=feature_y,
            title=f"{feature_x} vs {feature_y}",
            opacity=0.6,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "This relationship plot offers an additional view of numeric structure and potential redundancy, "
            "which can inform feature selection or transformation choices."
        )
        plots_rendered += 1

    st.subheader("1.4 Correlation Heatmap")
    if numeric_cols:
        corr = df[numeric_cols + [target_col]].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, ax=ax, cmap="coolwarm", center=0, linewidths=0.5)
        ax.set_title("Correlation Heatmap (Numerical Features)")
        st.pyplot(fig)

        corr_pairs = corr.unstack().dropna()
        corr_pairs = corr_pairs[corr_pairs.index.get_level_values(0) != corr_pairs.index.get_level_values(1)]
        strongest_pos = corr_pairs.idxmax()
        strongest_neg = corr_pairs.idxmin()
        st.caption(
            f"Strongest positive correlation: `{strongest_pos[0]}` with `{strongest_pos[1]}` "
            f"(r={corr_pairs[strongest_pos]:.2f}). Strongest negative correlation: "
            f"`{strongest_neg[0]}` with `{strongest_neg[1]}` (r={corr_pairs[strongest_neg]:.2f}). "
            "High absolute correlations suggest potential multicollinearity or redundant features, "
            "which may influence model stability and interpretability."
        )
    else:
        st.info("Not enough numeric features available to compute a correlation heatmap.")


def render_metrics_block(result: dict, task_type: str):
    metrics = result["metrics"]
    if task_type == "classification":
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        c2.metric("Precision", f"{metrics['precision']:.3f}")
        c3.metric("Recall", f"{metrics['recall']:.3f}")
        c4.metric("F1", f"{metrics['f1']:.3f}")
        if metrics.get("roc_auc") is not None:
            c5.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
        else:
            c5.metric("ROC-AUC", "n/a")

        plot_confusion_matrix(result["y_true"], result["y_pred"])
        if result.get("y_score") is not None:
            plot_roc_curve(result["y_true"], result["y_score"])
        else:
            st.info("ROC curve not available because the model does not output probabilities.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"{metrics['mae']:.3f}")
        c2.metric("RMSE", f"{metrics['rmse']:.3f}")
        c3.metric("R2", f"{metrics['r2']:.3f}")
        plot_predicted_vs_actual(result["y_true"], result["y_pred"])


def render_part2_predictive(df: pd.DataFrame):
    st.header("Part 2: Predictive Analytics")

    target_col = TARGET_COLUMN if TARGET_COLUMN in df.columns else df.columns[-1]
    task_override = None if TASK_TYPE == "auto" else TASK_TYPE
    task_type = detect_task_type(df[target_col], task_override)

    st.subheader("2.1 Data Preparation")
    st.markdown(
        f"""
        This section prepares the dataset for predictive modeling. The target variable is `{target_col}`.
        The task type is detected as **{task_type}** using the target distribution and data type.
        A 70/30 train/test split is used with `random_state=42`, and preprocessing is fit only on the training set
        to avoid data leakage.
        """
    )

    if len(df) < 10:
        st.warning("The dataset is very small. Modeling results may be unstable.")

    col_types = get_column_types(df, target_col)
    preprocessor = build_preprocessor(col_types["numeric"], col_types["categorical"], dense_output=False)
    preprocessor_dense = build_preprocessor(col_types["numeric"], col_types["categorical"], dense_output=True)

    if preprocessor is None:
        st.error("No usable features found after excluding the target column.")
        return

    X_train, X_test, y_train, y_test = prepare_data(df, target_col, task_type)

    st.markdown(
        "Preprocessing steps applied: median imputation for numeric features, most-frequent imputation for "
        "categorical features, standard scaling for numeric columns, and one-hot encoding for categorical columns."
    )
    st.caption(f"Numeric columns: {', '.join(col_types['numeric']) or 'None'}")
    st.caption(f"Categorical columns: {', '.join(col_types['categorical']) or 'None'}")

    shapes_table = pd.DataFrame(
        [
            {"Split": "X_train", "Rows": X_train.shape[0], "Columns": X_train.shape[1]},
            {"Split": "X_test", "Rows": X_test.shape[0], "Columns": X_test.shape[1]},
            {"Split": "y_train", "Rows": y_train.shape[0], "Columns": 1},
            {"Split": "y_test", "Rows": y_test.shape[0], "Columns": 1},
        ]
    )
    st.table(shapes_table)

    cv = get_cv_strategy(task_type, y_train)
    if cv is None:
        st.warning("Not enough data to run 5-fold cross-validation; some models will be skipped.")

    results = []

    st.subheader("2.2 Linear or Logistic Regression Baseline")
    with st.spinner("Training baseline model..."):
        baseline = train_baseline_model(task_type, preprocessor, X_train, y_train, X_test, y_test)
    if "y_true" not in baseline:
        baseline["y_true"] = y_test
    results.append(baseline)
    st.markdown(f"Baseline model: **{baseline['model_name']}**")
    render_metrics_block(baseline, task_type)
    st.markdown(
        "This baseline establishes a linear benchmark for comparison. It provides a simple, interpretable "
        "reference point before introducing nonlinear models."
    )

    st.subheader("2.3 Decision Tree / CART")
    if cv is None:
        st.warning("Decision tree skipped due to insufficient data for cross-validation.")
    else:
        with st.spinner("Tuning decision tree with 5-fold CV..."):
            tree_result = train_decision_tree_model(task_type, preprocessor, X_train, y_train, X_test, y_test, cv)
        if tree_result is None:
            st.warning("Decision tree skipped.")
        else:
            if "y_true" not in tree_result:
                tree_result["y_true"] = y_test
            results.append(tree_result)
            st.markdown(f"Best hyperparameters: `{tree_result['best_params']}`")
            st.markdown(f"Best CV score: `{tree_result['cv_best_score']:.4f}`")
            render_metrics_block(tree_result, task_type)

            model_step = tree_result["model"].named_steps["model"]
            if model_step.get_depth() <= 4:
                fig, ax = plt.subplots(figsize=(10, 6))
                try:
                    feature_names = tree_result["model"].named_steps["preprocess"].get_feature_names_out()
                except Exception:
                    feature_names = None
                plot_tree(model_step, feature_names=feature_names, filled=True, ax=ax)
                st.pyplot(fig)

            st.markdown(
                "Decision trees capture nonlinear rules and interaction effects while remaining interpretable. "
                "The tuned tree illustrates how a small set of splits can approximate complex relationships."
            )

    st.subheader("2.4 Random Forest")
    if cv is None:
        st.warning("Random forest skipped due to insufficient data for cross-validation.")
    else:
        with st.spinner("Tuning random forest with 5-fold CV..."):
            rf_result = train_random_forest_model(task_type, preprocessor, X_train, y_train, X_test, y_test, cv)
        if rf_result is None:
            st.warning("Random forest skipped.")
        else:
            if "y_true" not in rf_result:
                rf_result["y_true"] = y_test
            results.append(rf_result)
            st.markdown(f"Best hyperparameters: `{rf_result['best_params']}`")
            st.markdown(f"Best CV score: `{rf_result['cv_best_score']:.4f}`")
            render_metrics_block(rf_result, task_type)

            model_step = rf_result["model"].named_steps["model"]
            try:
                feature_names = rf_result["model"].named_steps["preprocess"].get_feature_names_out()
                importances = model_step.feature_importances_
                imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
                imp_df = imp_df.sort_values("importance", ascending=False).head(15)
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.barh(imp_df["feature"], imp_df["importance"])
                ax.set_title("Top Feature Importances")
                ax.invert_yaxis()
                st.pyplot(fig)
            except Exception:
                st.info("Feature importance could not be computed for this model.")

            st.markdown(
                "Random forests average many trees to reduce variance and improve generalization. "
                "The feature importance plot highlights which inputs drive predictions most strongly."
            )

    st.subheader("2.5 Boosted Trees - XGBoost or LightGBM")
    if cv is None:
        st.warning("Boosted trees skipped due to insufficient data for cross-validation.")
    else:
        with st.spinner("Tuning boosted trees with 5-fold CV..."):
            boosted_result = train_boosted_model(
                task_type, preprocessor, preprocessor_dense, X_train, y_train, X_test, y_test, cv
            )
        if boosted_result is None:
            st.warning("Boosted trees skipped.")
        else:
            if "y_true" not in boosted_result:
                boosted_result["y_true"] = y_test
            results.append(boosted_result)
            if "Fallback" in boosted_result["model_name"]:
                st.warning("XGBoost/LightGBM not available. Using sklearn HistGradientBoosting as fallback.")
            st.markdown(f"Model used: **{boosted_result['model_name']}**")
            st.markdown(f"Best hyperparameters: `{boosted_result['best_params']}`")
            st.markdown(f"Best CV score: `{boosted_result['cv_best_score']:.4f}`")
            render_metrics_block(boosted_result, task_type)
            st.markdown(
                "Boosted trees sequentially correct errors from prior trees, often yielding strong predictive "
                "performance at the cost of interpretability and higher training time."
            )

    st.subheader("2.6 Neural Network - MLP")
    if preprocessor_dense is None:
        st.warning("MLP skipped because no usable features were available.")
    else:
        with st.spinner("Training neural network..."):
            mlp_result = train_mlp_model(task_type, preprocessor_dense, X_train, y_train, X_test, y_test)
        if mlp_result is None:
            st.warning("MLP skipped.")
        else:
            if "y_true" not in mlp_result:
                mlp_result["y_true"] = y_test
            results.append(mlp_result)
            if not mlp_result.get("used_tf", False):
                st.warning("TensorFlow not available. Using sklearn MLP as fallback.")
            render_metrics_block(mlp_result, task_type)

            history = mlp_result.get("history")
            if history is not None:
                hist_df = pd.DataFrame(history.history)
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(hist_df["loss"], label="train")
                if "val_loss" in hist_df:
                    ax.plot(hist_df["val_loss"], label="validation")
                ax.set_title("Training Loss")
                ax.set_xlabel("Epoch")
                ax.legend()
                st.pyplot(fig)

                if task_type == "classification" and "accuracy" in hist_df:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.plot(hist_df["accuracy"], label="train")
                    if "val_accuracy" in hist_df:
                        ax.plot(hist_df["val_accuracy"], label="validation")
                    ax.set_title("Training Accuracy")
                    ax.set_xlabel("Epoch")
                    ax.legend()
                    st.pyplot(fig)

            st.markdown(
                "Neural networks can model complex nonlinear patterns but require more tuning and data. "
                "Early stopping helps reduce overfitting by halting training when validation performance plateaus."
            )

    st.subheader("2.7 Model Comparison Summary")
    if not results:
        st.warning("No models were trained successfully.")
        return

    comparison = build_model_comparison_table(results, task_type)
    st.dataframe(comparison, use_container_width=True)

    if task_type == "classification":
        best_idx = comparison["f1"].idxmax()
        metric_name = "F1"
        key_metric = comparison["f1"]
        fig = px.bar(comparison, x="model", y="f1", title="F1 Score by Model")
    else:
        best_idx = comparison["rmse"].idxmin()
        metric_name = "RMSE"
        key_metric = comparison["rmse"]
        fig = px.bar(comparison, x="model", y="rmse", title="RMSE by Model (lower is better)")

    st.plotly_chart(fig, use_container_width=True)
    best_model_name = comparison.loc[best_idx, "model"]
    best_metric_value = key_metric.loc[best_idx]

    st.markdown(
        f"""
        The best-performing model on the held-out test set is **{best_model_name}** with {metric_name} =
        **{best_metric_value:.3f}**. This outcome aligns with expectations that more flexible models often
        outperform linear baselines, though the gains must be weighed against interpretability and training cost.
        Ensemble methods typically deliver strong accuracy but are harder to explain, while trees and linear models
        remain easier to interpret and deploy. Neural networks can be competitive when enough data and tuning are available.
        """
    )

    st.session_state["part2_results"] = results
    st.session_state["part2_context"] = {
        "task_type": task_type,
        "target_col": target_col,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def select_best_tree_model(results: list[dict], task_type: str):
    tree_models = []
    for result in results:
        name = result.get("model_name", "")
        if name.startswith("Random Forest") or name in {
            "XGBoost",
            "LightGBM",
            "HistGradientBoosting (Fallback)",
            "Boosted Trees (LightGBM)",
        }:
            tree_models.append(result)

    if not tree_models:
        return None, "No tree-based models were available from Part 2."

    if task_type == "classification":
        best = max(tree_models, key=lambda r: r["metrics"].get("f1", -1))
        metric_name = "F1"
        metric_value = best["metrics"].get("f1")
    else:
        best = min(tree_models, key=lambda r: r["metrics"].get("rmse", np.inf))
        metric_name = "RMSE"
        metric_value = best["metrics"].get("rmse")

    reason = f"Selected `{best['model_name']}` because it achieved the best {metric_name} on the test set ({metric_value:.3f})."
    return best, reason


def get_transformed_feature_names(preprocessor, X: pd.DataFrame) -> list[str]:
    if preprocessor is None:
        return list(X.columns)
    try:
        names = preprocessor.get_feature_names_out()
        return [str(n) for n in names]
    except Exception:
        return list(X.columns)


def compute_shap_values(model_pipeline, X_sample: pd.DataFrame, task_type: str):
    try:
        import shap  # type: ignore
    except Exception:
        return None, "SHAP is not installed. Install it with `pip install shap`."

    if hasattr(model_pipeline, "named_steps"):
        preprocessor = model_pipeline.named_steps.get("preprocess")
        model = model_pipeline.named_steps.get("model")
    else:
        preprocessor = None
        model = model_pipeline

    if preprocessor is not None:
        X_trans = preprocessor.transform(X_sample)
        feature_names = get_transformed_feature_names(preprocessor, X_sample)
    else:
        X_trans = X_sample.values
        feature_names = list(X_sample.columns)

    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()

    try:
        explainer = shap.TreeExplainer(model)
    except Exception as exc:
        return None, f"TreeExplainer could not be initialized for this model: {exc}"

    try:
        shap_values = explainer(X_trans)
        explanation = shap_values
    except Exception:
        shap_values = explainer.shap_values(X_trans)
        explanation = None

    shap_values_used = shap_values
    if isinstance(shap_values, list):
        if task_type == "classification" and len(shap_values) > 1:
            shap_values_used = shap_values[1]
        else:
            shap_values_used = shap_values[0]
    elif hasattr(shap_values, "values") and getattr(shap_values.values, "ndim", 1) > 2:
        if task_type == "classification" and shap_values.values.shape[1] > 1:
            shap_values_used = shap_values[:, 1, :]
        else:
            shap_values_used = shap_values[:, 0, :]

    return {
        "shap": shap,
        "explainer": explainer,
        "shap_values": shap_values_used,
        "explanation": explanation,
        "X_trans": X_trans,
        "feature_names": feature_names,
    }, None


def plot_shap_summary(shap_module, shap_values, X_trans, feature_names, task_type: str):
    try:
        shap_module.summary_plot(shap_values, X_trans, feature_names=feature_names, show=False)
        fig = plt.gcf()
        st.pyplot(fig)
        plt.close(fig)
    except Exception:
        st.info("Could not render the SHAP summary plot.")


def plot_shap_bar(shap_module, shap_values, X_trans, feature_names):
    try:
        shap_module.summary_plot(shap_values, X_trans, feature_names=feature_names, plot_type="bar", show=False)
        fig = plt.gcf()
        st.pyplot(fig)
        plt.close(fig)
    except Exception:
        st.info("Could not render the SHAP bar plot.")


def choose_interesting_observation(task_type: str, model_pipeline, X_test: pd.DataFrame, y_test: pd.Series):
    preds = model_pipeline.predict(X_test)
    if task_type == "classification":
        reason = "Selected the highest-probability positive prediction."
        idx = X_test.index[0]
        probs = None
        if hasattr(model_pipeline, "predict_proba"):
            probs = model_pipeline.predict_proba(X_test)
            if probs.ndim == 2 and probs.shape[1] == 2:
                score = probs[:, 1]
            else:
                score = probs.max(axis=1)
            idx = X_test.index[np.argmax(score)]
            reason = "Selected the highest predicted probability instance for the positive class."
        if y_test is not None:
            misclassified = X_test.index[preds != y_test]
            if len(misclassified) > 0 and probs is not None:
                idx = misclassified[0]
                reason = "Selected a misclassified example to highlight a challenging case."
        return idx, reason

    residuals = np.abs(preds - y_test)
    idx = X_test.index[np.argmax(residuals)]
    reason = "Selected the observation with the largest absolute residual."
    return idx, reason


def plot_shap_waterfall(shap_module, explainer, X_row_trans, feature_names):
    try:
        explanation = explainer(X_row_trans)
        if hasattr(explanation, "__getitem__"):
            explanation = explanation[0]
        shap_module.plots.waterfall(explanation, show=False, max_display=15)
        fig = plt.gcf()
        st.pyplot(fig)
        plt.close(fig)
    except Exception:
        st.info("Could not render the SHAP waterfall plot for this observation.")


def generate_shap_interpretation(shap_values, feature_names, task_type: str, local_reason: str):
    values = None
    if hasattr(shap_values, "values"):
        values = shap_values.values
    elif isinstance(shap_values, list):
        values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        values = shap_values

    if values is None:
        return (
            "Global feature effects could not be computed, but the SHAP plots above indicate which features drive predictions.",
            "Local explanation could not be generated.",
            "These insights can still help stakeholders focus on the features most tied to model behavior.",
        )

    if values.ndim > 2:
        values = values[:, 1, :] if values.shape[1] > 1 else values[:, 0, :]

    mean_abs = np.abs(values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:5]
    top_features = [feature_names[i] for i in top_idx]

    mean_signed = values.mean(axis=0)
    signed_desc = []
    for i in top_idx[:3]:
        direction = "positive" if mean_signed[i] >= 0 else "negative"
        signed_desc.append(f"{feature_names[i]} ({direction} influence)")

    global_text = (
        f"The SHAP summary indicates that the most influential features are {', '.join(top_features)}. "
        f"On average, the strongest directional effects are {', '.join(signed_desc)}, which suggests these "
        "features most consistently push predictions up or down."
    )

    local_text = (
        f"The waterfall plot explains a single prediction. {local_reason} The largest positive and negative "
        "contributors in that row show how the model combined specific feature values to reach its output."
    )

    decision_text = (
        "For decision-makers, these results highlight which factors most affect outcomes and which levers are "
        "likely to move predictions. This can guide targeted interventions, policy adjustments, or prioritization "
        "of data quality improvements for the most sensitive variables."
    )
    return global_text, local_text, decision_text


def render_part3_explainability():
    st.header("Part 3: Explainability")
    st.subheader("3.1 SHAP Analysis")

    if "part2_results" not in st.session_state or "part2_context" not in st.session_state:
        st.warning("Run Part 2 first so the models and test set are available for SHAP analysis.")
        return

    results = st.session_state["part2_results"]
    context = st.session_state["part2_context"]
    task_type = context["task_type"]
    X_train = context["X_train"]
    X_test = context["X_test"]
    y_test = context["y_test"]

    best_tree, reason = select_best_tree_model(results, task_type)
    if best_tree is None:
        st.warning(reason)
        return

    st.markdown(f"Selected model for SHAP: **{best_tree['model_name']}**")
    st.markdown(reason)

    sample_size = min(200, len(X_train))
    if sample_size < len(X_train):
        st.info(f"Using a reproducible sample of {sample_size} rows for SHAP to keep performance responsive.")
    sample = X_train.sample(n=sample_size, random_state=42)

    shap_payload, error = compute_shap_values(best_tree["model"], sample, task_type)
    if shap_payload is None:
        st.warning(error)
        return

    shap_module = shap_payload["shap"]
    shap_values = shap_payload["shap_values"]
    X_trans = shap_payload["X_trans"]
    feature_names = shap_payload["feature_names"]
    explainer = shap_payload["explainer"]

    st.markdown("**Summary plot (beeswarm)**")
    plot_shap_summary(shap_module, shap_values, X_trans, feature_names, task_type)
    st.caption("The beeswarm plot summarizes global feature impact and direction across the sample.")

    st.markdown("**Bar plot of mean absolute SHAP values**")
    plot_shap_bar(shap_module, shap_values, X_trans, feature_names)
    st.caption("The bar chart ranks features by average absolute SHAP value (global importance).")

    idx, local_reason = choose_interesting_observation(task_type, best_tree["model"], X_test, y_test)
    st.markdown(f"**Waterfall plot for observation `{idx}`**")
    st.caption(local_reason)

    if hasattr(best_tree["model"], "named_steps"):
        preprocess = best_tree["model"].named_steps.get("preprocess")
    else:
        preprocess = None

    X_row = X_test.loc[[idx]]
    if preprocess is not None:
        X_row_trans = preprocess.transform(X_row)
    else:
        X_row_trans = X_row.values
    if hasattr(X_row_trans, "toarray"):
        X_row_trans = X_row_trans.toarray()

    plot_shap_waterfall(shap_module, explainer, X_row_trans, feature_names)
    st.caption("The waterfall plot explains how individual features push the prediction above or below the baseline.")

    global_text, local_text, decision_text = generate_shap_interpretation(
        shap_values, feature_names, task_type, local_reason
    )
    st.markdown("**Interpretation**")
    st.markdown(global_text)
    st.markdown(local_text)
    st.markdown(decision_text)


@st.cache_resource
def load_saved_models():
    model_map = {
        "Linear Regression": MODEL_DIR / "linear.joblib",
        "Ridge Regression": MODEL_DIR / "ridge.joblib",
        "Lasso Regression": MODEL_DIR / "lasso.joblib",
        "Decision Tree (CART)": MODEL_DIR / "cart.joblib",
        "Random Forest": MODEL_DIR / "random_forest.joblib",
        "Boosted Trees (LightGBM)": MODEL_DIR / "lightgbm.joblib",
    }
    models = {}
    for name, path in model_map.items():
        if path.exists():
            try:
                models[name] = joblib.load(path)
            except Exception:
                continue
    return models


@st.cache_data
def load_artifacts():
    return {
        "metrics": load_json(ARTIFACT_DIR / "metrics.json"),
        "summary": load_json(ARTIFACT_DIR / "summary.json"),
        "diagnostics": load_json(ARTIFACT_DIR / "diagnostics.json"),
        "shap_index": load_json(ARTIFACT_DIR / "shap_index.json"),
    }


# NOTE: Do not use cache_data here because `models` is a dict of trained model objects,
# which is unhashable for Streamlit's data cache. Models are cached separately via
# `@st.cache_resource` in `load_saved_models`.
def compute_model_results(models: dict, X_test: pd.DataFrame, y_test: pd.Series, task_type: str):
    results = []
    for name, model in models.items():
        try:
            y_pred = model.predict(X_test)
        except Exception:
            continue

        y_score = None
        if task_type == "classification":
            if hasattr(model, "predict_proba"):
                try:
                    y_score = model.predict_proba(X_test)
                    if y_score.ndim == 2 and y_score.shape[1] == 2:
                        y_score = y_score[:, 1]
                except Exception:
                    y_score = None
            elif hasattr(model, "decision_function"):
                try:
                    y_score = model.decision_function(X_test)
                except Exception:
                    y_score = None
            metrics = evaluate_classification(y_test, y_pred, y_score)
        else:
            metrics = evaluate_regression(y_test, y_pred)

        results.append(
            {
                "model_name": name,
                "model": model,
                "metrics": metrics,
                "y_pred": y_pred,
                "y_score": y_score,
                "y_true": y_test,
            }
        )
    return results


def extract_model_hyperparams(model_pipeline):
    if not hasattr(model_pipeline, "named_steps"):
        return {}
    model_step = model_pipeline.named_steps.get("model")
    if model_step is None:
        return {}
    params = model_step.get_params()
    keys = [
        "n_estimators",
        "max_depth",
        "min_samples_leaf",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "alpha",
        "l1_ratio",
        "lambda",
    ]
    return {k: params[k] for k in keys if k in params}


def render_executive_summary_tab(df: pd.DataFrame, summary: dict | None, best_model_name: str | None):
    st.header("Executive Summary")
    st.markdown(
        f"""
        **Dataset:** {DATASET_DESCRIPTION}  
        **Prediction task:** {PREDICTION_TASK_DESCRIPTION}  
        **Why it matters:** {WHY_IT_MATTERS}
        """
    )
    st.info(APPROACH_SUMMARY)

    c1, c2, c3 = st.columns(3)
    c1.metric("Records", f"{len(df):,}")
    c2.metric("Features", f"{df.shape[1] - 1}")
    c3.metric("Target", TARGET_COLUMN)

    if best_model_name:
        st.success(f"Best model: {best_model_name}")

    st.markdown(KEY_FINDINGS_SUMMARY)
    if summary:
        st.caption("Artifacts are loaded.")


def render_descriptive_analytics_tab(df: pd.DataFrame):
    st.header("Descriptive Analytics")
    st.subheader("Target and Feature Overview")
    render_part1_descriptive(df)

    st.subheader("Additional Distributions")
    st.plotly_chart(eda.price_distribution(df), use_container_width=True)
    st.caption("Distribution of listing prices to understand scale and spread.")

    st.plotly_chart(eda.log_price_distribution(df), use_container_width=True)
    st.caption("Log-transformed prices to highlight variability in the tails.")

    c1, c2 = st.columns(2)
    c1.plotly_chart(eda.price_by_borough(df), use_container_width=True)
    c1.caption("Price varies by borough, indicating strong geographic effects.")
    c2.plotly_chart(eda.price_by_room_type(df), use_container_width=True)
    c2.caption("Room type drives distinct pricing tiers.")

    c3, c4 = st.columns(2)
    c3.plotly_chart(eda.listings_by_borough(df), use_container_width=True)
    c3.caption("Listing concentration by borough reveals supply hotspots.")
    c4.plotly_chart(eda.top_neighbourhoods_by_listing_count(df), use_container_width=True)
    c4.caption("Top neighborhoods highlight market density.")

    st.plotly_chart(eda.median_price_by_borough_room(df), use_container_width=True)
    st.caption("Median price splits by borough and room type show segmentation patterns.")

    st.plotly_chart(eda.map_scatter(df), use_container_width=True)
    st.caption("Geospatial scatter shows price clustering by location.")

    c5, c6 = st.columns(2)
    c5.plotly_chart(eda.minimum_nights_vs_price(df), use_container_width=True)
    c5.caption("Minimum nights are associated with different price bands.")
    c6.plotly_chart(eda.reviews_vs_price(df), use_container_width=True)
    c6.caption("Review activity provides a proxy for demand signals.")

    c7, c8 = st.columns(2)
    c7.plotly_chart(eda.availability_vs_price(df), use_container_width=True)
    c7.caption("Availability relates to price and potential occupancy strategies.")
    c8.plotly_chart(eda.review_activity_over_time(df), use_container_width=True)
    c8.caption("Seasonality in review activity can indicate demand cycles.")

    st.plotly_chart(eda.correlation_heatmap(df), use_container_width=True)
    st.caption("Correlation heatmap highlights linear relationships among numeric features.")


def render_model_performance_tab(results: list[dict], task_type: str):
    st.header("Model Performance")
    if not results:
        st.warning("No saved models found. Please ensure trained models exist in `artifacts/models`.")
        return

    comparison = build_model_comparison_table(results, task_type)
    st.subheader("Model Comparison Summary")
    st.dataframe(comparison, use_container_width=True)

    if task_type == "classification":
        metric_name = "F1"
        fig = px.bar(comparison, x="model", y="f1", title="F1 Score by Model")
        best_idx = comparison["f1"].idxmax()
    else:
        metric_name = "RMSE"
        fig = px.bar(comparison, x="model", y="rmse", title="RMSE by Model (lower is better)")
        best_idx = comparison["rmse"].idxmin()

    st.plotly_chart(fig, use_container_width=True)
    best_model_name = comparison.loc[best_idx, "model"]
    st.success(f"Best-performing model: {best_model_name} ({metric_name})")

    if task_type == "classification":
        fig, ax = plt.subplots(figsize=(6, 4))
        plotted = False
        for result in results:
            if result.get("y_score") is None:
                continue
            try:
                RocCurveDisplay.from_predictions(
                    result["y_true"], result["y_score"], ax=ax, name=result["model_name"]
                )
                plotted = True
            except Exception:
                continue
        if plotted:
            ax.set_title("ROC Curves by Model")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("ROC curves are unavailable for these models.")

    st.subheader("Performance Details")
    for result in results:
        with st.expander(result["model_name"]):
            render_metrics_block(result, task_type)
            params = extract_model_hyperparams(result["model"])
            if params:
                st.markdown("**Key hyperparameters**")
                st.json(params)
            else:
                st.caption("Hyperparameters not available or not applicable for this model.")

    st.markdown(
        "Overall, ensemble methods generally outperform linear baselines, but gains must be weighed against "
        "interpretability and deployment complexity."
    )


def select_interactive_features(df: pd.DataFrame, target_col: str, task_type: str):
    col_types = get_column_types(df, target_col)
    num_candidates = top_numeric_features(df, target_col, col_types["numeric"], task_type, k=6)
    if not num_candidates:
        num_candidates = col_types["numeric"][:6]
    cat_candidates = top_categorical_features(df, target_col, col_types["categorical"], k=4)
    if not cat_candidates:
        cat_candidates = col_types["categorical"][:4]
    return num_candidates, cat_candidates


def default_value_from_series(series: pd.Series):
    if series is None or series.empty:
        return None
    if pd.api.types.is_bool_dtype(series):
        mode = series.mode(dropna=True)
        return bool(mode.iloc[0]) if not mode.empty else False
    if pd.api.types.is_numeric_dtype(series):
        median = series.median(skipna=True)
        if pd.notna(median):
            return float(median)
        mean = series.mean(skipna=True)
        if pd.notna(mean):
            return float(mean)
        return 0.0
    if pd.api.types.is_datetime64_any_dtype(series):
        mode = series.mode(dropna=True)
        if not mode.empty:
            return mode.iloc[0]
        return pd.Timestamp.now()
    mode = series.mode(dropna=True)
    if not mode.empty:
        return mode.iloc[0]
    return "unknown"


def generic_default_for_name(col: str):
    lower = col.lower()
    if lower.startswith("is_") or lower.startswith("has_") or lower.endswith("_flag"):
        return False
    if any(token in lower for token in ["count", "num", "number", "qty", "quantity", "age", "days", "month", "year", "id", "price"]):
        return 0.0
    return "unknown"


def compute_engineered_defaults(df: pd.DataFrame) -> tuple[dict, dict]:
    defaults: dict[str, object] = {}
    context: dict[str, object] = {"max_review_date": None}
    if "last_review" in df.columns:
        reviews = pd.to_datetime(df["last_review"], errors="coerce")
        max_date = reviews.max()
        context["max_review_date"] = max_date if pd.notna(max_date) else None
        if pd.notna(max_date):
            days_since = (max_date - reviews).dt.days
        else:
            days_since = pd.Series([np.nan] * len(reviews))
        defaults["days_since_last_review"] = default_value_from_series(days_since)
        defaults["last_review_year"] = default_value_from_series(reviews.dt.year)
        defaults["last_review_month"] = default_value_from_series(reviews.dt.month)
        defaults["has_review"] = default_value_from_series(reviews.notna().astype(int))
    return defaults, context


def get_expected_input_columns(model, summary: dict | None, fallback_df: pd.DataFrame | None, target_col: str):
    if summary and isinstance(summary, dict):
        summary_cols = summary.get("input_features")
        if summary_cols:
            return list(summary_cols)
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if hasattr(model, "named_steps"):
        preprocess = model.named_steps.get("preprocess")
        if preprocess is not None and hasattr(preprocess, "feature_names_in_"):
            return list(preprocess.feature_names_in_)
        if preprocess is not None and hasattr(preprocess, "transformers_") and fallback_df is not None:
            cols: list[str] = []
            for _, _, col_spec in preprocess.transformers_:
                if col_spec is None:
                    continue
                if isinstance(col_spec, slice):
                    cols.extend(list(fallback_df.columns[col_spec]))
                elif isinstance(col_spec, (list, tuple, np.ndarray, pd.Index)):
                    cols.extend(list(col_spec))
                else:
                    try:
                        cols.extend(list(col_spec))
                    except Exception:
                        continue
            if cols:
                return cols
    if fallback_df is not None:
        return [c for c in fallback_df.columns if c != target_col]
    return []


def create_default_input_row(df: pd.DataFrame, target_col: str, expected_cols: list[str]):
    defaults: dict[str, object] = {}
    engineered_defaults, context = compute_engineered_defaults(df)
    for col in expected_cols:
        if col == target_col:
            continue
        if col in df.columns:
            defaults[col] = default_value_from_series(df[col])
        elif col in engineered_defaults:
            defaults[col] = engineered_defaults[col]
        else:
            defaults[col] = generic_default_for_name(col)
    return defaults, context


def build_interactive_input_form(
    df: pd.DataFrame,
    target_col: str,
    task_type: str,
    expected_cols: list[str],
    defaults: dict,
):
    numeric_cols, categorical_cols = select_interactive_features(df, target_col, task_type)
    numeric_cols = [c for c in numeric_cols if c in expected_cols]
    categorical_cols = [c for c in categorical_cols if c in expected_cols]
    user_values = {}

    st.markdown("**User-controlled features**")
    for col in numeric_cols:
        default_val = defaults.get(col, 0.0)
        if default_val is None or (isinstance(default_val, float) and np.isnan(default_val)):
            default_val = 0.0
        user_values[col] = st.number_input(col, value=float(default_val))
    for col in categorical_cols:
        options = sorted(df[col].dropna().astype(str).unique().tolist())
        default_val = defaults.get(col, "")
        if default_val is None:
            default_val = ""
        default_val = str(default_val)
        if default_val not in options and options:
            default_val = options[0]
        user_values[col] = st.selectbox(col, options, index=options.index(default_val) if default_val in options else 0)

    auto_filled = {k: v for k, v in defaults.items() if k not in user_values}
    st.caption("Auto-filled features use dataset means (numeric) or most frequent values (categorical).")
    st.dataframe(pd.DataFrame(auto_filled.items(), columns=["Feature", "Auto-filled value"]).head(10), use_container_width=True)
    return user_values


def apply_engineered_from_inputs(
    row: dict,
    user_values: dict,
    expected_cols: list[str],
    context: dict,
):
    if "last_review" not in user_values:
        return row
    last_review = pd.to_datetime(user_values.get("last_review"), errors="coerce")
    if "has_review" in expected_cols:
        row["has_review"] = 0 if pd.isna(last_review) else 1
    if "last_review_year" in expected_cols and pd.notna(last_review):
        row["last_review_year"] = int(last_review.year)
    if "last_review_month" in expected_cols and pd.notna(last_review):
        row["last_review_month"] = int(last_review.month)
    if "days_since_last_review" in expected_cols:
        max_date = context.get("max_review_date")
        if pd.notna(last_review) and pd.notna(max_date):
            row["days_since_last_review"] = max(int((max_date - last_review).days), 0)
        elif pd.notna(last_review):
            row["days_since_last_review"] = 0
    return row


def fill_missing_prediction_columns(
    input_df: pd.DataFrame,
    expected_cols: list[str],
    defaults: dict,
):
    if not expected_cols:
        return input_df
    missing = [c for c in expected_cols if c not in input_df.columns]
    for col in missing:
        input_df[col] = defaults.get(col, generic_default_for_name(col))
    extra = [c for c in input_df.columns if c not in expected_cols]
    if extra:
        input_df = input_df.drop(columns=extra)
    return input_df[expected_cols]


def create_input_dataframe(
    user_values: dict,
    defaults: dict,
    expected_cols: list[str],
    target_col: str,
    context: dict,
):
    full_row = {col: defaults.get(col, generic_default_for_name(col)) for col in expected_cols}
    for key, value in user_values.items():
        if key in full_row:
            full_row[key] = value
    full_row = apply_engineered_from_inputs(full_row, user_values, expected_cols, context)
    df_row = pd.DataFrame([full_row], columns=expected_cols)
    if target_col in df_row.columns:
        df_row = df_row.drop(columns=[target_col])
    return df_row


def predict_from_user_input(
    model,
    input_df: pd.DataFrame,
    task_type: str,
    expected_cols: list[str],
    defaults: dict,
):
    input_df = fill_missing_prediction_columns(input_df, expected_cols, defaults)
    pred = model.predict(input_df)
    if task_type == "classification":
        pred_label = pred[0]
        prob = None
        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba(input_df)
                if probs.ndim == 2 and probs.shape[1] == 2:
                    prob = float(probs[0, 1])
            except Exception:
                prob = None
        return pred_label, prob
    return float(pred[0]), None


def explain_custom_prediction_with_shap(model_pipeline, input_df: pd.DataFrame):
    try:
        import shap  # type: ignore
    except Exception:
        st.warning("SHAP is not installed. Install it with `pip install shap` to enable explanations.")
        return

    if hasattr(model_pipeline, "named_steps"):
        preprocess = model_pipeline.named_steps.get("preprocess")
        model = model_pipeline.named_steps.get("model")
    else:
        preprocess = None
        model = model_pipeline

    if preprocess is not None:
        X_trans = preprocess.transform(input_df)
        feature_names = get_transformed_feature_names(preprocess, input_df)
    else:
        X_trans = input_df.values
        feature_names = list(input_df.columns)

    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()

    try:
        explainer = shap.TreeExplainer(model)
        explanation = explainer(X_trans)
        if hasattr(explanation, "__getitem__"):
            explanation = explanation[0]
        shap.plots.waterfall(explanation, show=False, max_display=15)
        fig = plt.gcf()
        st.pyplot(fig)
        plt.close(fig)
    except Exception:
        st.info("SHAP waterfall plot is only available for supported tree-based models.")


def render_explainability_tab(
    df: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    task_type: str,
    results: list[dict],
    best_tree_result: dict | None,
    summary: dict | None,
):
    st.header("Explainability & Interactive Prediction")

    st.subheader("Explainability")
    if best_tree_result is None:
        st.warning("No tree-based model available for SHAP explainability.")
    else:
        sample_size = min(200, len(X_train))
        sample = X_train.sample(n=sample_size, random_state=42)
        shap_payload, error = compute_shap_values(best_tree_result["model"], sample, task_type)
        if shap_payload is None:
            st.warning(error)
        else:
            shap_module = shap_payload["shap"]
            shap_values = shap_payload["shap_values"]
            X_trans = shap_payload["X_trans"]
            feature_names = shap_payload["feature_names"]

            st.markdown("**SHAP Summary (Beeswarm)**")
            plot_shap_summary(shap_module, shap_values, X_trans, feature_names, task_type)
            st.caption("Beeswarm plot shows global feature impact and direction.")

            st.markdown("**SHAP Bar Plot**")
            plot_shap_bar(shap_module, shap_values, X_trans, feature_names)
            st.caption("Bar plot ranks features by mean absolute SHAP values.")

            global_text, _, decision_text = generate_shap_interpretation(
                shap_values, feature_names, task_type, "Selected a representative example."
            )
            st.markdown(global_text)
            st.markdown(decision_text)

    st.subheader("Interactive Prediction")
    models = load_saved_models()
    if not models:
        st.warning("No saved models available for prediction.")
        return

    model_name = st.selectbox("Select model for prediction", list(models.keys()))
    selected_model = models[model_name]
    expected_cols = get_expected_input_columns(selected_model, summary, df, TARGET_COLUMN)
    if not expected_cols:
        expected_cols = [c for c in df.columns if c != TARGET_COLUMN]
    defaults_source = df
    try:
        defaults_source = split_features_target(df).X
    except Exception:
        defaults_source = df
    defaults, context = create_default_input_row(defaults_source, TARGET_COLUMN, expected_cols)
    user_values = build_interactive_input_form(
        df,
        TARGET_COLUMN,
        task_type,
        expected_cols,
        defaults,
    )
    input_df = create_input_dataframe(
        user_values,
        defaults,
        expected_cols,
        TARGET_COLUMN,
        context,
    )

    prediction, prob = predict_from_user_input(
        selected_model,
        input_df,
        task_type,
        expected_cols,
        defaults,
    )
    if task_type == "classification":
        st.metric("Predicted class", prediction)
        if prob is not None:
            st.metric("Predicted probability (positive class)", f"{prob:.3f}")
    else:
        st.metric("Predicted value", f"{prediction:.2f}")

    st.markdown("**SHAP Waterfall for Custom Input**")
    tree_model_for_shap = selected_model
    if best_tree_result is not None and "Random Forest" not in model_name and "Boosted" not in model_name:
        st.caption("Selected model is not tree-based. Using best tree-based model for SHAP waterfall.")
        tree_model_for_shap = best_tree_result["model"]
    explain_custom_prediction_with_shap(tree_model_for_shap, input_df)


def metrics_table(metrics_payload: dict) -> pd.DataFrame:
    rows = []
    for model_name, payload in metrics_payload.items():
        rows.append(
            {
                "model": model_name,
                "cv_rmse": payload["cv"]["rmse_mean"],
                "cv_rmse_std": payload["cv"]["rmse_std"],
                "cv_mae": payload["cv"]["mae_mean"],
                "cv_r2": payload["cv"]["r2_mean"],
                "holdout_rmse": payload["holdout"]["rmse"],
                "holdout_mae": payload["holdout"]["mae"],
                "holdout_r2": payload["holdout"]["r2"],
                "train_seconds": payload["train_seconds"],
            }
        )
    return pd.DataFrame(rows).sort_values("cv_rmse")


def executive_summary(df: pd.DataFrame, summary: dict | None, metrics: dict | None):
    st.subheader("Executive Summary")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Listings", f"{len(df):,}")
    c2.metric("Features", f"{df.shape[1] - 1}")
    c3.metric("Mean Price", f"${df['price'].mean():.1f}")
    c4.metric("Median Price", f"${df['price'].median():.1f}")
    c5.metric("Price Std", f"${df['price'].std():.1f}")

    borough = df["neighbourhood_group"].value_counts().idxmax()
    room = df["room_type"].value_counts().idxmax()

    if summary and metrics:
        best = summary.get("best_model", "n/a")
        rmse = metrics[best]["cv"]["rmse_mean"]
        r2 = metrics[best]["cv"]["r2_mean"]
        st.markdown(
            f"""
            - Best cross-validated model: `{best}`
            - Best model CV RMSE: `{rmse:.2f}`
            - Best model CV R2: `{r2:.3f}`
            - Most represented borough: `{borough}`
            - Most common room type: `{room}`
            """
        )
    else:
        st.info("Run `python build_artifacts.py` in `archive/` to populate metrics, SHAP artifacts, and trained models.")


def shap_tab_content(summary: dict | None, shap_index: dict | None):
    st.subheader("SHAP Explainability")
    if not summary or not shap_index:
        st.info("SHAP artifacts missing. Run `python build_artifacts.py` first.")
        return

    available_models = [m for m in summary["models_trained"] if m in shap_index]
    model_name = st.selectbox("Model for SHAP analysis", available_models)

    values, transformed_data, importance = load_shap_payload(model_name)
    if values is None:
        st.warning("Could not load SHAP arrays for this model.")
        return

    plot_summary = PLOT_DIR / f"shap_summary_{model_name}.png"
    plot_bar = PLOT_DIR / f"shap_bar_{model_name}.png"

    left, right = st.columns(2)
    if plot_summary.exists():
        left.image(str(plot_summary), caption=f"SHAP Beeswarm - {model_name}")
    if plot_bar.exists():
        right.image(str(plot_bar), caption=f"SHAP Global Importance - {model_name}")

    st.markdown("Top SHAP features (mean absolute SHAP):")
    st.dataframe(importance.head(20), use_container_width=True)

    top_features = importance["feature"].head(30).tolist()
    feature = st.selectbox("Dependence feature", top_features)
    feature_index = transformed_data.columns.get_loc(feature)

    dep_fig = px.scatter(
        x=transformed_data[feature],
        y=values[:, feature_index],
        opacity=0.55,
        labels={"x": feature, "y": "SHAP value"},
        title=f"SHAP Dependence Plot - {feature}",
    )
    st.plotly_chart(dep_fig, use_container_width=True)

    row_idx = st.slider("Inspect single prediction index", 0, len(transformed_data) - 1, 0)
    contrib = (
        pd.DataFrame({"feature": transformed_data.columns, "shap_value": values[row_idx]})
        .assign(abs_val=lambda d: d["shap_value"].abs())
        .sort_values("abs_val", ascending=False)
        .head(15)
    )

    local_fig = px.bar(
        contrib.sort_values("shap_value"),
        x="shap_value",
        y="feature",
        orientation="h",
        title="Local Explanation (Top 15 contributions)",
    )
    st.plotly_chart(local_fig, use_container_width=True)


st.title("NYC Airbnb Open Data: Complete Modeling Studio")

artifacts = load_artifacts()
metrics = artifacts["metrics"]
summary = artifacts["summary"]
diagnostics = artifacts["diagnostics"]

df = load_dataset()
target_col = TARGET_COLUMN if TARGET_COLUMN in df.columns else df.columns[-1]
task_override = None if TASK_TYPE == "auto" else TASK_TYPE
task_type = detect_task_type(df[target_col], task_override)
X_train, X_test, y_train, y_test = prepare_data(df, target_col, task_type)

models = load_saved_models()
results = compute_model_results(models, X_test, y_test, task_type)
best_tree_result, _ = select_best_tree_model(results, task_type) if results else (None, "")
best_model_name = None
if results:
    comp = build_model_comparison_table(results, task_type)
    if task_type == "classification":
        best_model_name = comp.loc[comp["f1"].idxmax(), "model"]
    else:
        best_model_name = comp.loc[comp["rmse"].idxmin(), "model"]

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Executive Summary",
        "Descriptive Analytics",
        "Model Performance",
        "Explainability & Interactive Prediction",
    ]
)

with tab1:
    render_executive_summary_tab(df, summary, best_model_name)
    st.caption("Model artifacts should live in `artifacts/models` and include pipelines with preprocessing.")

with tab2:
    render_descriptive_analytics_tab(df)

with tab3:
    render_model_performance_tab(results, task_type)

with tab4:
    render_explainability_tab(df, X_train, X_test, task_type, results, best_tree_result, summary)
