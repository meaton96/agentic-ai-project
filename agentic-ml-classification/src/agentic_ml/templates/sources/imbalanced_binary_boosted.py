"""
Recipe template: HistGradientBoostingClassifier tuned for imbalanced
binary targets, using sklearn's native categorical support (no
one-hot) and built-in class_weight="balanced" reweighting instead of
resampling. Resampling techniques (e.g. SMOTE) are deliberately not
used here: synthetic-sample generation is easy to get wrong across CV
folds (fitting the resampler before the split boundary is a classic
leakage bug — see priors/general/leakage_rules.md Rule 5) and
class_weight reweighting achieves a similar effect without that risk.

Use when the profiler reports is_imbalanced=true.

config keys:
  numeric_cols        (required) list[str]
  categorical_cols     (required) list[str]
  class_weight         (optional, default "balanced")
  max_iter             (optional, default 300) — boosting rounds
  learning_rate        (optional, default 0.1)
  seed                 (optional, default 42) — harness-controlled.
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


def build_pipeline(config: dict):
    numeric_cols = config["numeric_cols"]
    categorical_cols = config["categorical_cols"]
    seed = config.get("seed", 42)

    preprocessor = ColumnTransformer([
        ("numeric", "passthrough", numeric_cols),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            # unseen-at-inference categories map to NaN (not a fabricated
            # integer category) because HistGradientBoostingClassifier's
            # native categorical support requires missing values to be
            # NaN, not an out-of-range integer code.
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)),
        ]), categorical_cols),
    ])

    # ColumnTransformer output column order matches transformer
    # declaration order above: numeric_cols then categorical_cols.
    categorical_mask = [False] * len(numeric_cols) + [True] * len(categorical_cols)

    clf = HistGradientBoostingClassifier(
        class_weight=config.get("class_weight", "balanced"),
        max_iter=config.get("max_iter", 300),
        learning_rate=config.get("learning_rate", 0.1),
        categorical_features=categorical_mask,
        random_state=seed,
    )

    return Pipeline([("pre", preprocessor), ("clf", clf)])
