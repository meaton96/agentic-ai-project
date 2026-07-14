"""
Deterministic pre-target dataset facts + DatasetSpec proposal validation.

This is what makes natural-language (or dataset-only) intake possible
without ever letting an LLM invent a column name: the IntakeAgent only
ever sees column facts this module computed (raw_schema_summary), and
the harness re-validates its proposed DatasetSpec fields against those
same facts (validate_dataset_spec_proposal) before profiler/modeling
ever run. Same trust boundary as harness/profiler.py, one step earlier:
here there isn't even a target_column yet, so nothing target-dependent
(imbalance, correlation-based leakage flags) can be computed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from .profiler import DATETIME_NAME_HINTS, GROUP_NAME_HINTS, ID_NAME_HINTS, _looks_like_datetime, _name_hints

# Upper bound on distinct target values accepted as classification labels.
# Above this, a column is far more likely a continuous/ID field mistakenly
# pointed at as the target than a genuine set of class labels.
MAX_CLASSES = 20


@dataclass
class RawColumnSummary:
    name: str
    dtype: str
    missing_frac: float
    n_unique: int
    unique_frac: float
    sample_values: list
    name_suggests_id: bool
    name_suggests_group: bool
    looks_like_datetime: bool

    def to_dict(self) -> dict:
        return asdict(self)


def raw_schema_summary(df: pd.DataFrame) -> dict:
    """Column facts computable with NO target column declared — the only
    thing the intake agent is allowed to see before it proposes one."""
    n_rows = len(df)
    columns = []
    for col in df.columns:
        series = df[col]
        n_unique = int(series.nunique(dropna=True))
        columns.append(RawColumnSummary(
            name=col,
            dtype=str(series.dtype),
            missing_frac=round(float(series.isna().mean()), 4) if n_rows else 0.0,
            n_unique=n_unique,
            unique_frac=round(n_unique / n_rows, 4) if n_rows else 0.0,
            sample_values=series.dropna().head(3).astype(str).tolist(),
            name_suggests_id=_name_hints(col, ID_NAME_HINTS),
            name_suggests_group=_name_hints(col, GROUP_NAME_HINTS),
            looks_like_datetime=_looks_like_datetime(series) or _name_hints(col, DATETIME_NAME_HINTS),
        ).to_dict())
    return {"n_rows": n_rows, "n_columns": len(df.columns), "columns": columns}


def validate_dataset_spec_proposal(df: pd.DataFrame, proposal: dict) -> list[str]:
    """Structural + classification-eligibility validation of an intake
    agent's proposed DatasetSpec fields. Does not trust the agent's
    'task' claim — always requires the target to have between 2 and
    MAX_CLASSES non-null unique values (binary or multiclass; see
    priors/general — MVP scope)."""
    if not isinstance(proposal, dict):
        return ["top-level response is not a JSON object"]

    errors: list[str] = []
    target = proposal.get("target_column")
    if not target or target not in df.columns:
        errors.append(f"target_column '{target}' not found in dataset columns")
        return errors  # nothing else is checkable without a valid target

    n_unique_target = int(df[target].dropna().nunique())
    if n_unique_target < 2 or n_unique_target > MAX_CLASSES:
        errors.append(
            f"target_column '{target}' has {n_unique_target} unique non-null values; "
            f"this pipeline requires between 2 and {MAX_CLASSES} distinct class labels"
        )

    for field_name in ("group_column", "time_column"):
        val = proposal.get(field_name)
        if val is None:
            continue
        if val not in df.columns:
            errors.append(f"{field_name} '{val}' not found in dataset columns")
        if val == target:
            errors.append(f"{field_name} cannot be the same as target_column")

    id_columns = proposal.get("id_columns") or []
    if not isinstance(id_columns, list):
        errors.append("id_columns must be a list")
    else:
        for col in id_columns:
            if col not in df.columns:
                errors.append(f"id_columns references unknown column '{col}'")
            if col == target:
                errors.append(f"id_columns cannot include the target_column '{col}'")

    return errors
