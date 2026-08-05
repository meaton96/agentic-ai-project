"""
Compacts a wide per-column fact list into fewer entries when many
columns share a `{prefix}__{suffix}` naming pattern — exactly the shape
harness/timeseries_features.py's rollup produces (e.g. 23 sensors x 12
stats = 276 columns named "volt1__mean", "volt1__std", ...). Serializing
one full entry per column blew a 32K-token local model's context window
on a 282-column rolled-up NGAFID table (~19-25K tokens for the raw
schema / profile alone, before anything else in the prompt) — this is
what keeps that payload bounded regardless of how many derived stat
columns a rollup produces, without losing which real columns exist or
the deterministic facts about them.

Grouping happens ONLY at serialization (called from
intake.py::raw_schema_summary and profiler.py::ProfileReport.to_dict),
never inside the underlying fact computation — every per-column loop in
those modules still runs over every real column, so leakage/imbalance/
split-strategy logic (which needs every column's true value) is never
affected by what an agent is shown afterward. expand_grouped_columns()
is the inverse: callers that need to resolve a specific column name
back to its facts (steps/modeling_step.py, harness/feature_engineering.py's
config validators) use it instead of assuming a flat name-keyed dict,
since a compacted payload no longer is one.
"""
from __future__ import annotations

from collections import defaultdict

# Below this many members, a shared prefix is left ungrouped — avoids
# collapsing two or three columns that coincidentally share a "__"-split
# prefix into a group for no real payload-size benefit.
DEFAULT_MIN_GROUP_SIZE = 4


def group_columns_by_pattern(columns: list[dict], min_group_size: int = DEFAULT_MIN_GROUP_SIZE) -> list[dict]:
    """`columns`: a list of per-column fact dicts, each with a 'name' key
    (ColumnProfileEntry.to_dict() or raw_schema_summary()'s column shape —
    this only reads 'name' and passes every other field through
    unexamined, so it works for either shape). Columns whose name
    matches "{prefix}__{suffix}" and share a prefix with at least
    min_group_size total members are collapsed into one compact entry;
    every other column passes through unchanged. Original order is
    preserved; a grouped entry appears at its first member's position.
    A dataset with no "__"-suffixed columns (every dataset this pipeline
    has run against before the aviation rollup) returns `columns`
    unchanged."""
    groups: dict[str, list[dict]] = defaultdict(list)
    prefix_of: dict[str, str] = {}
    for c in columns:
        name = c["name"]
        if "__" in name:
            prefix, _, _ = name.rpartition("__")
            groups[prefix].append(c)
            prefix_of[name] = prefix

    grouped_prefixes = {p for p, members in groups.items() if len(members) >= min_group_size}
    if not grouped_prefixes:
        return list(columns)

    out: list[dict] = []
    emitted_prefixes: set[str] = set()
    for c in columns:
        prefix = prefix_of.get(c["name"])
        if prefix not in grouped_prefixes:
            out.append(c)
            continue
        if prefix in emitted_prefixes:
            continue  # this group's compact entry was already emitted
        members = groups[prefix]
        missing_fracs = [m["missing_frac"] for m in members if m.get("missing_frac") is not None]
        out.append({
            "column_group": prefix,
            "n_columns": len(members),
            "columns": [m["name"] for m in members],
            "missing_frac_max": max(missing_fracs) if missing_fracs else None,
            "representative_column": members[0],
        })
        emitted_prefixes.add(prefix)
    return out


def expand_grouped_columns(columns: list[dict]) -> dict[str, dict]:
    """Inverse of group_columns_by_pattern: {real_column_name: fact_dict},
    covering every real column whether or not it was compacted into a
    group. A group's members all resolve to its representative_column's
    facts (dtype/is_likely_id/etc.) — safe because compaction only groups
    columns sharing a stat-family naming pattern, which are homogeneous
    by construction (harness/timeseries_features.py's rollup produces
    every {sensor}__{stat} column the same deterministic way)."""
    resolved: dict[str, dict] = {}
    for c in columns:
        if "column_group" in c:
            rep = c["representative_column"]
            for name in c["columns"]:
                resolved[name] = rep
        else:
            resolved[c["name"]] = c
    return resolved
