"""
Recipe template registry: Layer 2 of the priors hierarchy
(priors/general/baseline_ladder.md and the architecture doc's "Layer 2:
Parameterized recipe templates"). Each template is a verified, static
.py file under templates/sources/ exposing build_pipeline(config) per
the harness/sandbox.py contract — the modeling agent's job is to pick
one template_id and fill in its config "holes" (column lists, a few
hyperparameters), never to write pipeline code from scratch.

The registry is the only place template_id -> source file mapping is
defined, and the only place required/optional config keys are declared
for validation before a candidate ever reaches the sandbox.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SOURCES_DIR = Path(__file__).parent / "sources"


@dataclass(frozen=True)
class TemplateSpec:
    template_id: str
    title: str
    description: str
    when_to_use: str
    required_config_keys: list[str]
    optional_config_keys: dict[str, Any]  # key -> default value, for the agent's reference
    source_filename: str

    def source_path(self) -> Path:
        return _SOURCES_DIR / self.source_filename

    def read_source(self) -> str:
        return self.source_path().read_text()

    def to_summary_dict(self) -> dict:
        return {
            "template_id": self.template_id,
            "title": self.title,
            "description": self.description,
            "when_to_use": self.when_to_use,
            "required_config_keys": self.required_config_keys,
            "optional_config_keys": self.optional_config_keys,
        }


TEMPLATES: dict[str, TemplateSpec] = {
    spec.template_id: spec
    for spec in [
        TemplateSpec(
            template_id="logistic_numeric",
            title="Numeric-only logistic regression baseline",
            description=(
                "Median-impute + scale numeric features, then LogisticRegression. "
                "Categorical columns are ignored entirely."
            ),
            when_to_use=(
                "No (or negligible) categorical columns, or you want the cheapest "
                "possible check for rough linear separability."
            ),
            required_config_keys=["numeric_cols"],
            optional_config_keys={"C": 1.0, "penalty": "l2", "max_iter": 1000},
            source_filename="logistic_numeric.py",
        ),
        TemplateSpec(
            template_id="sklearn_mixed_pipeline",
            title="Mixed numeric/categorical sklearn pipeline",
            description=(
                "ColumnTransformer (median-impute+scale numeric, most_frequent-"
                "impute+one-hot categorical) feeding a configurable classifier: "
                "logistic_regression or random_forest."
            ),
            when_to_use=(
                "General-purpose default for mixed tabular data with low-to-"
                "moderate cardinality categoricals (roughly <=30 unique values)."
            ),
            required_config_keys=["numeric_cols", "categorical_cols"],
            optional_config_keys={"classifier": "logistic_regression", "C": 1.0, "n_estimators": 300},
            source_filename="sklearn_mixed_pipeline.py",
        ),
        TemplateSpec(
            template_id="lightgbm_mixed",
            title="LightGBM with native categorical handling",
            description=(
                "Casts categorical columns to pandas 'category' dtype so LightGBM "
                "splits on them natively (no one-hot); numeric NaNs are left "
                "un-imputed since LightGBM routes missing values during training."
            ),
            when_to_use=(
                "Categorical columns have moderate-to-high cardinality, or a "
                "stronger non-linear baseline than HistGradientBoosting is wanted."
            ),
            required_config_keys=["numeric_cols", "categorical_cols"],
            optional_config_keys={"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31},
            source_filename="lightgbm_mixed.py",
        ),
        TemplateSpec(
            template_id="xgboost_mixed",
            title="XGBoost with one-hot categoricals and native missing handling",
            description=(
                "Numeric columns pass through un-imputed (XGBoost routes NaN "
                "natively); categorical columns are most_frequent-imputed then "
                "one-hot encoded (portable across xgboost>=2.0.0 without relying "
                "on version-specific enable_categorical behavior)."
            ),
            when_to_use=(
                "Want a strong gradient-boosted baseline and are comfortable with "
                "one-hot-encoded categoricals (roughly <=30 unique values)."
            ),
            required_config_keys=["numeric_cols", "categorical_cols"],
            optional_config_keys={"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1},
            source_filename="xgboost_mixed.py",
        ),
        TemplateSpec(
            template_id="imbalanced_binary_boosted",
            title="HistGradientBoosting for imbalanced binary targets",
            description=(
                "class_weight='balanced' reweighting (not resampling — avoids the "
                "classic fit-resampler-before-split leakage bug) plus sklearn's "
                "native categorical_features support, no one-hot."
            ),
            when_to_use="Profiler reports is_imbalanced=true.",
            required_config_keys=["numeric_cols", "categorical_cols"],
            optional_config_keys={"class_weight": "balanced", "max_iter": 300, "learning_rate": 0.1},
            source_filename="imbalanced_binary_boosted.py",
        ),
        TemplateSpec(
            template_id="high_cardinality_target_encoding",
            title="Cross-fitted target encoding for high-cardinality categoricals",
            description=(
                "sklearn's built-in TargetEncoder (internally cross-fitted, "
                "leakage-safe by construction) for categorical columns, plus "
                "median-impute+scale numeric features, feeding a configurable "
                "classifier: hist_gradient_boosting or logistic_regression."
            ),
            when_to_use=(
                "categorical_cols include high-cardinality columns (profiler "
                "n_unique above roughly 30) where one-hot would blow up "
                "dimensionality."
            ),
            required_config_keys=["numeric_cols", "categorical_cols"],
            optional_config_keys={"classifier": "hist_gradient_boosting"},
            source_filename="high_cardinality_target_encoding.py",
        ),
    ]
}


def get_template(template_id: str) -> TemplateSpec:
    if template_id not in TEMPLATES:
        raise KeyError(f"Unknown template_id '{template_id}'. Available: {sorted(TEMPLATES)}")
    return TEMPLATES[template_id]


def list_template_summaries() -> list[dict]:
    return [t.to_summary_dict() for t in TEMPLATES.values()]


def validate_config(template_id: str, config: dict) -> list[str]:
    """Structural validation only (required keys present, list-typed column
    fields) — this does NOT check that column names actually exist in the
    dataset; that cross-check happens against the profiler report in the
    modeling agent script, since the registry has no access to the dataset."""
    template = get_template(template_id)
    errors = []
    for key in template.required_config_keys:
        if key not in config:
            errors.append(f"missing required config key: '{key}'")
    for key in ("numeric_cols", "categorical_cols"):
        if key in config and not isinstance(config[key], list):
            errors.append(f"config key '{key}' must be a list")
    return errors
