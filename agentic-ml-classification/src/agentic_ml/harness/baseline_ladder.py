"""
Baseline ladder used to evaluate a dataset with zero agent/LLM involvement.
This is Phase 1 of the build plan and should work standalone.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from lightgbm import LGBMClassifier
    _HAS_LIGHTGBM = True
except ImportError:
    _HAS_LIGHTGBM = False

try:
    from xgboost import XGBClassifier
    _HAS_XGBOOST = True
except ImportError:
    _HAS_XGBOOST = False


@dataclass
class ColumnProfile:
    numeric_cols: list[str]
    categorical_cols: list[str]

    @classmethod
    def infer(cls, X: pd.DataFrame, max_categorical_cardinality: int = 50) -> "ColumnProfile":
        numeric_cols, categorical_cols = [], []
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                numeric_cols.append(col)
            elif X[col].nunique(dropna=True) <= max_categorical_cardinality:
                categorical_cols.append(col)
            else:
                # high-cardinality categorical: still bucketed as categorical
                # for the baseline ladder; later phases can add target
                # encoding via a verified template instead.
                categorical_cols.append(col)
        return cls(numeric_cols=numeric_cols, categorical_cols=categorical_cols)


def _preprocessor(profile: ColumnProfile) -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, profile.numeric_cols),
        ("categorical", categorical_pipeline, profile.categorical_cols),
    ])


def build_baseline_pipelines(
    profile: ColumnProfile, seed: int = 42
) -> dict[str, Pipeline]:
    pre = _preprocessor(profile)
    pipelines = {
        "dummy_most_frequent": Pipeline([
            ("pre", pre),
            ("clf", DummyClassifier(strategy="most_frequent")),
        ]),
        "logistic_regression": Pipeline([
            ("pre", pre),
            ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
        ]),
        "random_forest": Pipeline([
            ("pre", pre),
            ("clf", RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)),
        ]),
        "hist_gradient_boosting": Pipeline([
            ("pre", pre),
            ("clf", HistGradientBoostingClassifier(random_state=seed)),
        ]),
    }
    if _HAS_LIGHTGBM:
        pipelines["lightgbm"] = Pipeline([
            ("pre", pre),
            ("clf", LGBMClassifier(random_state=seed, verbosity=-1)),
        ])
    if _HAS_XGBOOST:
        pipelines["xgboost"] = Pipeline([
            ("pre", pre),
            ("clf", XGBClassifier(
                random_state=seed, eval_metric="logloss", use_label_encoder=False,
                verbosity=0,
            )),
        ])
    return pipelines


def fit_and_predict(
    pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series,
    X_eval: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (y_pred, y_proba_positive_class)."""
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_eval)
    proba = pipeline.predict_proba(X_eval)
    # assume binary classification, positive class is the second column
    y_proba = proba[:, 1] if proba.shape[1] == 2 else proba.max(axis=1)
    return y_pred, y_proba
