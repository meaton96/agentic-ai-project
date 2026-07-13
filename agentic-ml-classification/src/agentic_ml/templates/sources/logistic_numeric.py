"""
Recipe template: numeric-only logistic regression baseline.

When to use: the profiler reports no (or negligible) categorical
columns, or you want the cheapest possible check for rough linear
separability before trying anything heavier. See
priors/general/baseline_ladder.md rung 2 — this template lets the
modeling agent reproduce that rung deliberately, with its own choice
of numeric_cols/C/penalty, instead of always falling back to the fixed
baseline ladder.

config keys:
  numeric_cols   (required) list[str] — columns to use as features.
  C              (optional, default 1.0) inverse regularization strength.
  penalty        (optional, default "l2").
  max_iter       (optional, default 1000).
  seed           (optional, default 42) — harness overwrites this at
                 build time regardless of what the agent proposes.
"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_pipeline(config: dict):
    numeric_cols = config["numeric_cols"]

    preprocessor = ColumnTransformer(
        [("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric_cols)],
        remainder="drop",
    )

    clf = LogisticRegression(
        C=config.get("C", 1.0),
        penalty=config.get("penalty", "l2"),
        max_iter=config.get("max_iter", 1000),
        random_state=config.get("seed", 42),
    )

    return Pipeline([("pre", preprocessor), ("clf", clf)])
