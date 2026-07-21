"""
Recipe template: sklearn's built-in cross-fitted TargetEncoder for
high-cardinality categorical columns (added in scikit-learn 1.3; this
repo requires scikit-learn>=1.4.0). TargetEncoder internally cross-fits
during .fit_transform(): each row's encoded value comes from a model
trained on the *other* folds of the training data, so a fold never
sees its own label baked into its own encoded feature. That is what
makes it safe to drop into a plain Pipeline step — no manual
leave-one-out or CV wrapping needed to avoid the classic
target-encoding leakage bug (priors/general/leakage_rules.md Rule 5).

Use when categorical_cols include columns with high cardinality
(profiler-reported n_unique above roughly 30) where one-hot encoding
would blow up dimensionality.

config keys:
  numeric_cols        (required) list[str]
  categorical_cols     (required) list[str]
  classifier           (optional, default "hist_gradient_boosting")
                        one of "hist_gradient_boosting" | "logistic_regression"
  seed                 (optional, default 42) — harness-controlled.
"""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder


def build_pipeline(config: dict):
    numeric_cols = config["numeric_cols"]
    categorical_cols = config["categorical_cols"]
    seed = config.get("seed", 42)

    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric_cols),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("target_encode", TargetEncoder(random_state=seed)),
        ]), categorical_cols),
    ])

    classifier_choice = config.get("classifier", "hist_gradient_boosting")
    if classifier_choice == "hist_gradient_boosting":
        clf = HistGradientBoostingClassifier(random_state=seed)
    elif classifier_choice == "logistic_regression":
        clf = LogisticRegression(max_iter=1000, random_state=seed)
    else:
        raise ValueError(f"unknown classifier '{classifier_choice}'")

    return Pipeline([("pre", preprocessor), ("clf", clf)])
