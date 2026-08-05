"""
Deterministic dataset profiling. This is the trusted, testable core —
the ProfilerAgent (in agentic_ml.tools) only narrates and recommends
based on what this module computes. No LLM call can change these facts.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

from .column_grouping import group_columns_by_pattern
from .leakage import check_suspicious_feature_correlation

DATETIME_NAME_HINTS = ("date", "time", "timestamp", "day", "created", "updated", "dob")
# A column named e.g. "date_diff" or "time_since_signup" contains a
# datetime-name token but holds a relative OFFSET (an int/float delta,
# often negative), not an absolute point in time — sorting groups by it
# (group_time strategy) sorts by that delta, not by calendar time, which
# can accidentally sort by whatever the delta correlates with (e.g. the
# label itself). These hints veto the name-hint match below; they don't
# affect _looks_like_datetime's content-based detection.
RELATIVE_OFFSET_NAME_HINTS = ("diff", "delta", "elapsed", "duration", "since", "until", "ago")
ID_NAME_HINTS = ("id", "uuid", "guid", "key", "index", "idx")
GROUP_NAME_HINTS = ("customer", "user", "patient", "machine", "session", "account", "entity", "device")

_TOKEN_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+")


@dataclass
class ColumnProfileEntry:
    name: str
    dtype: str
    missing_count: int
    missing_frac: float
    n_unique: int
    unique_frac: float
    is_likely_id: bool
    is_likely_categorical: bool
    is_likely_numeric: bool
    is_likely_datetime: bool
    is_likely_group_candidate: bool
    sample_values: list

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ProfileReport:
    n_rows: int
    n_columns: int
    target_column: str
    target_distribution: dict
    class_imbalance_ratio: float
    is_imbalanced: bool
    columns: list[ColumnProfileEntry]
    likely_id_columns: list[str]
    likely_datetime_columns: list[str]
    likely_group_columns: list[str]
    recommended_split_strategy: str
    recommended_split_reasoning: str
    leakage_risk_flags: list[str]

    def to_dict(self) -> dict:
        d = asdict(self)
        full_columns = [c.to_dict() if isinstance(c, ColumnProfileEntry) else c for c in self.columns]
        # Grouping happens ONLY here, at serialization — every computation
        # above (leakage_risk_flags, likely_group_columns, imbalance, ...)
        # already ran over the full per-column list; this can't affect any
        # of that, only what an agent (or a validator reading this dict —
        # see column_grouping.py's docstring) sees afterward.
        d["columns"] = group_columns_by_pattern(full_columns)
        return d


def _looks_like_datetime(series: pd.Series, sample_size: int = 50) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if series.dtype != object:
        return False
    sample = series.dropna().head(sample_size)
    if len(sample) == 0:
        return False
    try:
        with warnings.catch_warnings():
            # dateutil warns on inconsistent formats when probing arbitrary
            # object columns (e.g. "Ticket", "Name") that obviously aren't
            # dates — that's exactly the case this function exists to rule
            # out, so the warning is expected noise, not a real problem.
            warnings.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(sample, errors="coerce")
        return parsed.notna().mean() > 0.9
    except (ValueError, TypeError):
        return False


def _name_hints(name: str, hints: tuple[str, ...]) -> bool:
    """Matches hints against whole name tokens (snake_case/camelCase-aware),
    not a raw substring — a naive substring check flagged columns like
    'SepalWidthCm'/'PetalWidthCm' as ID columns because "id" is a
    substring of "Width", which broke every feature column on the Iris
    dataset."""
    tokens = {t.lower() for t in _TOKEN_RE.findall(name)}
    return any(h in tokens for h in hints)


def profile_dataset(
    df: pd.DataFrame,
    target_column: str,
    id_cardinality_threshold: float = 0.95,
    categorical_cardinality_max: int = 30,
    imbalance_threshold: float = 0.20,
    correlation_leak_threshold: float = 0.98,
) -> ProfileReport:
    n_rows = len(df)
    columns: list[ColumnProfileEntry] = []
    likely_id_columns: list[str] = []
    likely_datetime_columns: list[str] = []
    likely_group_columns: list[str] = []

    for col in df.columns:
        series = df[col]
        missing_count = int(series.isna().sum())
        missing_frac = missing_count / n_rows if n_rows else 0.0
        n_unique = int(series.nunique(dropna=True))
        unique_frac = n_unique / n_rows if n_rows else 0.0

        is_numeric = pd.api.types.is_numeric_dtype(series)
        is_float_dtype = pd.api.types.is_float_dtype(series)
        is_datetime = _looks_like_datetime(series) or (
            _name_hints(col, DATETIME_NAME_HINTS)
            and not _name_hints(col, RELATIVE_OFFSET_NAME_HINTS)
        )

        # High cardinality alone only implies "ID" for integer/string columns.
        # Continuous float measurements (Fare, Age, temperature, ...) are
        # almost always near-100% unique too, but that's a property of being
        # continuous, not of being an identifier — so cardinality-based ID
        # detection is restricted to non-float dtypes. Name hints (id, uuid,
        # ...) still apply regardless of dtype.
        cardinality_suggests_id = unique_frac >= id_cardinality_threshold and not is_float_dtype
        is_id = (
            col != target_column
            and (cardinality_suggests_id or _name_hints(col, ID_NAME_HINTS))
            and not is_datetime
        )
        is_categorical = (
            not is_id
            and not is_datetime
            and (not is_numeric or n_unique <= categorical_cardinality_max)
            and n_unique <= categorical_cardinality_max
        )
        # group candidate: name suggests an entity, AND values repeat
        # (unlike a true row-unique ID) — the defining trait of a group key
        is_group_candidate = (
            col != target_column
            and not is_datetime
            and _name_hints(col, GROUP_NAME_HINTS + ID_NAME_HINTS)
            and 1 < n_unique < n_rows
        )

        entry = ColumnProfileEntry(
            name=col,
            dtype=str(series.dtype),
            missing_count=missing_count,
            missing_frac=round(missing_frac, 4),
            n_unique=n_unique,
            unique_frac=round(unique_frac, 4),
            is_likely_id=is_id,
            is_likely_categorical=is_categorical,
            is_likely_numeric=is_numeric and not is_categorical,
            is_likely_datetime=is_datetime,
            is_likely_group_candidate=is_group_candidate,
            sample_values=series.dropna().head(3).astype(str).tolist(),
        )
        columns.append(entry)

        if is_id:
            likely_id_columns.append(col)
        if is_datetime:
            likely_datetime_columns.append(col)
        if is_group_candidate:
            likely_group_columns.append(col)

    # target distribution + imbalance
    target_counts = df[target_column].value_counts(normalize=True)
    target_distribution = {str(k): round(float(v), 4) for k, v in target_counts.items()}
    if len(target_counts) >= 2:
        imbalance_ratio = float(target_counts.min() / target_counts.max())
    else:
        imbalance_ratio = 1.0
    is_imbalanced = imbalance_ratio < imbalance_threshold

    # recommended split strategy, purely rule-based
    has_group = len(likely_group_columns) > 0
    has_time = len(likely_datetime_columns) > 0
    if has_group and has_time:
        strategy = "group_time"
        reasoning = (
            f"Group-like column(s) {likely_group_columns} and datetime-like column(s) "
            f"{likely_datetime_columns} detected. Use group_time to prevent both entity "
            f"and temporal leakage."
        )
    elif has_group:
        strategy = "group"
        reasoning = f"Group-like column(s) {likely_group_columns} detected with repeated values. Use group split to prevent entity leakage."
    elif has_time:
        strategy = "time"
        reasoning = f"Datetime-like column(s) {likely_datetime_columns} detected. Use time split to prevent temporal leakage."
    else:
        strategy = "stratified"
        reasoning = "No group or datetime columns detected. Stratified random split is appropriate."

    # leakage risk flags: reuse the correlation check against non-id,
    # non-datetime feature columns
    feature_cols = [
        c.name for c in columns
        if not c.is_likely_id and not c.is_likely_datetime and c.name != target_column
    ]
    leakage_flags: list[str] = []
    if feature_cols:
        corr_check = check_suspicious_feature_correlation(
            df[feature_cols], df[target_column], threshold=correlation_leak_threshold
        )
        if not corr_check.passed:
            leakage_flags.append(corr_check.detail)

    high_missing = [c.name for c in columns if c.missing_frac > 0.5 and c.name != target_column]
    if high_missing:
        leakage_flags.append(
            f"columns with >50% missing values (consider excluding via id_columns): {high_missing}"
        )

    return ProfileReport(
        n_rows=n_rows,
        n_columns=len(df.columns),
        target_column=target_column,
        target_distribution=target_distribution,
        class_imbalance_ratio=round(imbalance_ratio, 4),
        is_imbalanced=is_imbalanced,
        columns=columns,
        likely_id_columns=likely_id_columns,
        likely_datetime_columns=likely_datetime_columns,
        likely_group_columns=likely_group_columns,
        recommended_split_strategy=strategy,
        recommended_split_reasoning=reasoning,
        leakage_risk_flags=leakage_flags,
    )