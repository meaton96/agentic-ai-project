"""
Recipe template: LightGBM with native categorical handling.

Rather than one-hot encoding categoricals (which explodes dimensionality
for anything above single-digit cardinality), this template casts
categorical columns to pandas 'category' dtype and lets LightGBM's
sklearn API detect and split on them natively. Numeric NaNs are also
left as-is: LightGBM routes missing values during tree construction
rather than needing them imputed first. Use when categorical columns
have moderate-to-high cardinality, or when a stronger non-linear
baseline than sklearn's HistGradientBoostingClassifier is wanted.

The dtype cast is done via FunctionTransformer(pd.DataFrame.astype, ...)
rather than a candidate-defined transformer class. The unfitted pipeline
this function returns crosses a process boundary via pickle (see
harness/sandbox.py), and a class/function defined inside candidate.py
itself cannot be reliably resolved by pickle on the other side of that
boundary — its __module__ resolves to "candidate", which isn't a real
importable module, and re-importing candidate.py by coincidence (if one
exists on sys.path) yields a distinct, non-identical object. Composing
the pipeline entirely out of objects that already live in real,
importable libraries (pandas/sklearn/lightgbm here) sidesteps that
entirely. Every recipe template must follow this same rule.

config keys:
  numeric_cols        (required) list[str]
  categorical_cols     (required) list[str]
  n_estimators         (optional, default 300)
  learning_rate        (optional, default 0.05)
  num_leaves           (optional, default 31)
  seed                 (optional, default 42) — harness-controlled.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from lightgbm import LGBMClassifier


def build_pipeline(config: dict):
    numeric_cols = config["numeric_cols"]
    categorical_cols = config["categorical_cols"]
    seed = config.get("seed", 42)

    preprocessor = ColumnTransformer(
        [
            ("numeric", "passthrough", numeric_cols),
            ("categorical", "passthrough", categorical_cols),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    cast_categorical = FunctionTransformer(
        pd.DataFrame.astype,
        kw_args={"dtype": {col: "category" for col in categorical_cols}},
    )

    clf = LGBMClassifier(
        n_estimators=config.get("n_estimators", 300),
        learning_rate=config.get("learning_rate", 0.05),
        num_leaves=config.get("num_leaves", 31),
        random_state=seed,
        verbosity=-1,
    )

    return Pipeline([
        ("pre", preprocessor),
        ("cast_categorical", cast_categorical),
        ("clf", clf),
    ])
