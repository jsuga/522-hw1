from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

try:
    from lightgbm import LGBMRegressor

    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

TARGET_COL = "price"
DROP_COLS = ["id", "name", "host_id", "host_name"]
DATE_COL = "last_review"
TREE_MODELS = {"cart", "random_forest", "lightgbm"}
LINEAR_MODELS = {"linear", "lasso", "ridge"}


@dataclass
class Dataset:
    X: pd.DataFrame
    y: pd.Series
    feature_cols: List[str]


def load_data(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
        max_date = df[DATE_COL].max()
        if pd.notna(max_date):
            df["days_since_last_review"] = (max_date - df[DATE_COL]).dt.days
        else:
            df["days_since_last_review"] = np.nan
        df["last_review_year"] = df[DATE_COL].dt.year
        df["last_review_month"] = df[DATE_COL].dt.month
        df["has_review"] = df[DATE_COL].notna().astype(int)
        df = df.drop(columns=[DATE_COL])
    return df


def split_features_target(df: pd.DataFrame) -> Dataset:
    df = df.copy()
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df = engineer_features(df)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found")

    y = df[TARGET_COL].astype(float)
    X = df.drop(columns=[TARGET_COL])
    return Dataset(X=X, y=y, feature_cols=list(X.columns))


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = [c for c in X.columns if X[c].dtype == "object"]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )


def build_pipeline(model, X: pd.DataFrame) -> Pipeline:
    return Pipeline(steps=[("preprocess", make_preprocessor(X)), ("model", model)])


def get_model_zoo(include_lightgbm: bool = True) -> Dict[str, object]:
    models: Dict[str, object] = {
        "linear": LinearRegression(),
        "lasso": Lasso(alpha=0.01, max_iter=25000),
        "ridge": Ridge(alpha=1.0),
        "cart": DecisionTreeRegressor(
            max_depth=18,
            min_samples_leaf=3,
            random_state=42,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=220,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
    }

    if include_lightgbm and HAS_LGBM:
        models["lightgbm"] = LGBMRegressor(
            n_estimators=320,
            learning_rate=0.03,
            num_leaves=63,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
        )

    return models


def evaluate_regression(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def cross_validate_model(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, float | List[float]]:
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )

    rmse = -scores["test_rmse"]
    mae = -scores["test_mae"]
    r2 = scores["test_r2"]

    return {
        "n_splits": int(n_splits),
        "rmse_mean": float(np.mean(rmse)),
        "rmse_std": float(np.std(rmse)),
        "mae_mean": float(np.mean(mae)),
        "mae_std": float(np.std(mae)),
        "r2_mean": float(np.mean(r2)),
        "r2_std": float(np.std(r2)),
        "fold_rmse": [float(v) for v in rmse],
        "fold_mae": [float(v) for v in mae],
        "fold_r2": [float(v) for v in r2],
    }


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    feature_names: List[str] = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "num":
            feature_names.extend(list(cols))
        elif name == "cat":
            ohe = transformer.named_steps["onehot"]
            feature_names.extend(list(ohe.get_feature_names_out(cols)))
    return feature_names


def choose_best_model(metrics_payload: Dict[str, Dict]) -> str:
    return min(metrics_payload, key=lambda m: metrics_payload[m]["cv"]["rmse_mean"])


def save_json(path: str | Path, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
