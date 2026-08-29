"""
Deterministic feature-engineering operations. Layer 1 "tested
components" in the priors hierarchy — same trust model as the
modeling templates: the feature-engineering agent picks operations
from this vetted catalog and fills in parameters; it never writes
transformation code itself.

Every operation here is intentionally stateless and row-wise (depends
only on that row's own values, never a fitted statistic like a mean or
quantile) — that's what makes it safe to apply to the whole dataset
before the train/val/test split even exists, the same way the
profiler's own descriptive facts are computed dataset-wide. Anything
that needs a fitted statistic (imputation values, target encoding,
scaling) belongs inside a modeling template's ColumnTransformer
instead, fit only on the training fold — this module does not do that
kind of work, and the feature-engineering agent is not asked to either.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from agentic_ml.ablation import AblationConfig

from .column_grouping import expand_grouped_columns

FEATURE_OPS: dict[str, dict] = {
    "ratio": {
        "description": "Adds col_a / col_b as a new numeric column (division by zero yields NaN, not an error).",
        "required_params": ["col_a", "col_b"],
        "optional_params": {"new_column_name": None},
        "applicable_dtype": "numeric",
    },
    "interaction": {
        "description": "Adds col_a * col_b as a new numeric column.",
        "required_params": ["col_a", "col_b"],
        "optional_params": {"new_column_name": None},
        "applicable_dtype": "numeric",
    },
    "log1p": {
        "description": "Adds log1p(col) for a right-skewed, non-negative numeric column (negative values yield NaN).",
        "required_params": ["col"],
        "optional_params": {"new_column_name": None},
        "applicable_dtype": "numeric",
    },
    "datetime_parts": {
        "description": "Extracts one or more of year/month/day/dayofweek from a datetime-like column as new numeric columns.",
        "required_params": ["col", "parts"],
        "optional_params": {},
        "applicable_dtype": "datetime",
    },
    "missing_indicator": {
        "description": "Adds a binary column (1 = the original value was missing) for any column.",
        "required_params": ["col"],
        "optional_params": {"new_column_name": None},
        "applicable_dtype": "any",
    },
}

VALID_DATETIME_PARTS = {"year", "month", "day", "dayofweek"}

_NUMERIC_DTYPE_PREFIXES = ("int", "float", "uint")


def _is_numeric_dtype(dtype_str: str) -> bool:
    """Whether a column's pandas dtype supports arithmetic (ratio/interaction/
    log1p) — deliberately NOT the same question as the profiler's
    is_likely_numeric flag, which conflates "numeric dtype" with "should this
    be one-hot-encoded vs scaled in a baseline modeling pipeline". A
    low-cardinality integer count (e.g. SibSp, Parch) is categorical for that
    modeling-heuristic purpose but perfectly valid to add/multiply/divide —
    family_size = SibSp + Parch + 1 is a standard engineered feature, and
    gating on is_likely_numeric incorrectly rejected exactly this case in
    real usage. Validation here is about "would this operation raise or
    produce garbage", not a modeling-quality judgment call."""
    return dtype_str.lower().startswith(_NUMERIC_DTYPE_PREFIXES)


def list_feature_ops() -> list[dict]:
    return [{"op_id": op_id, **meta} for op_id, meta in FEATURE_OPS.items()]


def apply_ratio(df: pd.DataFrame, col_a: str, col_b: str, new_column_name: str | None = None) -> tuple[pd.DataFrame, list[str]]:
    name = new_column_name or f"{col_a}_over_{col_b}"
    df = df.copy()
    denominator = df[col_b].replace(0, np.nan)
    df[name] = df[col_a] / denominator
    return df, [name]


def apply_interaction(df: pd.DataFrame, col_a: str, col_b: str, new_column_name: str | None = None) -> tuple[pd.DataFrame, list[str]]:
    name = new_column_name or f"{col_a}_times_{col_b}"
    df = df.copy()
    df[name] = df[col_a] * df[col_b]
    return df, [name]


def apply_log1p(df: pd.DataFrame, col: str, new_column_name: str | None = None) -> tuple[pd.DataFrame, list[str]]:
    name = new_column_name or f"log1p_{col}"
    df = df.copy()
    non_negative = df[col].where(df[col] >= 0)
    df[name] = np.log1p(non_negative)
    return df, [name]


def apply_datetime_parts(df: pd.DataFrame, col: str, parts: list[str]) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    parsed = pd.to_datetime(df[col], errors="coerce")
    new_names = []
    for part in parts:
        name = f"{col}_{part}"
        df[name] = parsed.dt.dayofweek if part == "dayofweek" else getattr(parsed.dt, part)
        new_names.append(name)
    return df, new_names


def apply_missing_indicator(df: pd.DataFrame, col: str, new_column_name: str | None = None) -> tuple[pd.DataFrame, list[str]]:
    name = new_column_name or f"{col}_was_missing"
    df = df.copy()
    df[name] = df[col].isna().astype(int)
    return df, [name]


_DISPATCH = {
    "ratio": apply_ratio,
    "interaction": apply_interaction,
    "log1p": apply_log1p,
    "datetime_parts": apply_datetime_parts,
    "missing_indicator": apply_missing_indicator,
}


def apply_feature_op(df: pd.DataFrame, op_id: str, params: dict) -> tuple[pd.DataFrame, list[str]]:
    """Dispatches to the right apply_* function, returning (new_df, new_column_names).
    Raises KeyError for an unknown op_id; callers (feature_engineering_step.py)
    are expected to have already run validate_feature_proposal() first."""
    if op_id not in _DISPATCH:
        raise KeyError(f"Unknown feature op '{op_id}'. Available: {sorted(_DISPATCH)}")
    return _DISPATCH[op_id](df, **params)


def validate_feature_proposal(
    profile_report: dict,
    target_column: str,
    group_column: str | None,
    time_column: str | None,
    proposal: dict,
    ablation: Optional[AblationConfig] = None,
) -> list[str]:
    """Structural validation of a feature-engineering agent's proposal
    against the profiler's facts — mirrors modeling_step.py's
    _validate_candidate_columns. group_column/time_column may not be
    dropped (the split manager needs them) but ARE valid op inputs
    (e.g. datetime_parts on the time_column is the expected use case).

    ablation: research-only, see agentic_ml.ablation — every flag
    defaults to False, so ablation=None is identical to omitting it."""
    ablation = ablation or AblationConfig()
    if not isinstance(proposal, dict):
        return ["top-level response is not a JSON object"]

    errors: list[str] = []
    # expand_grouped_columns(), not a flat {c["name"]: c} comprehension —
    # profile_report["columns"] may contain compacted group entries (see
    # harness/column_grouping.py) for a wide rolled-up table, and a
    # proposed column name needs to resolve correctly either way.
    known_cols = expand_grouped_columns(profile_report["columns"])
    existing_names = set(known_cols)

    protected_from_drop = {target_column}
    if group_column:
        protected_from_drop.add(group_column)
    if time_column:
        protected_from_drop.add(time_column)

    drop_columns = proposal.get("drop_columns")
    if drop_columns is None:
        drop_columns = []
    if not isinstance(drop_columns, list):
        errors.append("drop_columns must be a list")
        drop_columns = []
    for col in drop_columns:
        if col not in known_cols:
            errors.append(f"drop_columns references unknown column '{col}'")
        elif not ablation.skip_protected_drop_check and col in protected_from_drop:
            errors.append(f"drop_columns cannot include '{col}' (target/group/time column)")

    derived = proposal.get("derived_features")
    if derived is None:
        derived = []
    if not isinstance(derived, list):
        errors.append("derived_features must be a list")
        derived = []

    proposed_new_names: set[str] = set()
    for i, feat in enumerate(derived):
        if not isinstance(feat, dict) or "op_id" not in feat:
            errors.append(f"derived_features[{i}] must be an object with an 'op_id'")
            continue
        op_id = feat["op_id"]
        if not ablation.skip_op_id_check and op_id not in FEATURE_OPS:
            errors.append(f"derived_features[{i}] references unknown op_id '{op_id}'")
            continue

        spec = FEATURE_OPS.get(op_id)
        if spec is None:
            # only reachable with skip_op_id_check active on a genuinely
            # unknown op_id — nothing left to validate params/dtype
            # against; this proposal will reach apply_feature_op() and
            # raise KeyError there instead of failing here with a message.
            continue
        params = feat.get("params") or {}
        if not isinstance(params, dict):
            errors.append(f"derived_features[{i}] ({op_id}) 'params' must be an object")
            continue

        for key in spec["required_params"]:
            if key not in params:
                errors.append(f"derived_features[{i}] ({op_id}) missing required param '{key}'")

        input_cols = [params[k] for k in ("col", "col_a", "col_b") if k in params]
        for col in input_cols:
            if not ablation.skip_target_column_check and col == target_column:
                errors.append(f"derived_features[{i}] ({op_id}) must not use the target_column '{col}' as input")
                continue
            if col not in known_cols:
                errors.append(f"derived_features[{i}] ({op_id}) references unknown column '{col}'")
                continue
            entry = known_cols[col]
            if (
                spec["applicable_dtype"] == "numeric"
                and not ablation.skip_numeric_dtype_check
                and not _is_numeric_dtype(entry["dtype"])
            ):
                errors.append(f"derived_features[{i}] ({op_id}) requires a numeric column; '{col}' is not")
            if (
                spec["applicable_dtype"] == "datetime"
                and not ablation.skip_datetime_dtype_check
                and not entry["is_likely_datetime"]
            ):
                errors.append(f"derived_features[{i}] ({op_id}) requires a datetime-like column; '{col}' is not")

        if op_id == "datetime_parts":
            parts = params.get("parts")
            if not isinstance(parts, list) or not parts:
                errors.append(f"derived_features[{i}] (datetime_parts) requires a non-empty 'parts' list")
            else:
                bad_parts = [p for p in parts if p not in VALID_DATETIME_PARTS]
                if bad_parts:
                    errors.append(
                        f"derived_features[{i}] (datetime_parts) has invalid parts {bad_parts}; "
                        f"valid: {sorted(VALID_DATETIME_PARTS)}"
                    )

        new_name = params.get("new_column_name") if isinstance(params, dict) else None
        if new_name:
            if new_name in existing_names or new_name in proposed_new_names:
                errors.append(
                    f"derived_features[{i}] new_column_name '{new_name}' collides with an "
                    "existing column or another proposed feature"
                )
            proposed_new_names.add(new_name)

    return errors
