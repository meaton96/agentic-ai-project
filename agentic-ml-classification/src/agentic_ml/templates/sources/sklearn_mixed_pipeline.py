"""
Recipe template: general-purpose mixed numeric/categorical sklearn
pipeline. Use as the default choice for mixed tabular data with
low-to-moderate cardinality categoricals (roughly <=30 unique values;
for higher cardinality prefer high_cardinality_target_encoding or
lightgbm_mixed, which don't one-hot-explode).

config keys:
  numeric_cols        (required) list[str]
  categorical_cols     (required) list[str]
  classifier           (optional, default "logistic_regression")
                        one of "logistic_regression" | "random_forest"
  C                    (optional, default 1.0) — used when classifier="logistic_regression"
  n_estimators         (optional, default 300) — used when classifier="random_forest"
  seed                 (optional, default 42) — harness-controlled.
"""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_cols),
    ])

    classifier_choice = config.get("classifier", "logistic_regression")
    if classifier_choice == "logistic_regression":
        clf = LogisticRegression(C=config.get("C", 1.0), max_iter=1000, random_state=seed)
    elif classifier_choice == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=config.get("n_estimators", 300), random_state=seed, n_jobs=-1,
        )
    else:
        raise ValueError(
            f"unknown classifier '{classifier_choice}'; expected "
            "'logistic_regression' or 'random_forest'"
        )

    return Pipeline([("pre", preprocessor), ("clf", clf)])
