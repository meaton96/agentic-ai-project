"""
Recipe template: XGBoost with one-hot-encoded categoricals and native
missing-value handling for numeric features.

Numeric columns are passed through un-imputed: XGBoost natively routes
missing values to the best branch at each split during training, so
median-imputing them first would only destroy signal ("value is
missing" can itself be predictive, e.g. a skipped survey question).
Categorical columns are imputed (most_frequent) then one-hot encoded —
XGBoost's native categorical support requires enable_categorical and
version-specific dtype handling that isn't guaranteed to be portable
across the xgboost>=2.0.0 range this pipeline targets, so one-hot is
the safer default here (LightGBM's native path is used instead in
lightgbm_mixed for that case).

config keys:
  numeric_cols        (required) list[str]
  categorical_cols     (required) list[str]
  n_estimators         (optional, default 300)
  max_depth            (optional, default 6)
  learning_rate        (optional, default 0.1)
  seed                 (optional, default 42) — harness-controlled.
"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


def build_pipeline(config: dict):
    numeric_cols = config["numeric_cols"]
    categorical_cols = config["categorical_cols"]
    seed = config.get("seed", 42)

    preprocessor = ColumnTransformer([
        ("numeric", "passthrough", numeric_cols),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_cols),
    ])

    clf = XGBClassifier(
        n_estimators=config.get("n_estimators", 300),
        max_depth=config.get("max_depth", 6),
        learning_rate=config.get("learning_rate", 0.1),
        random_state=seed,
        eval_metric="logloss",
        verbosity=0,
    )

    return Pipeline([("pre", preprocessor), ("clf", clf)])
