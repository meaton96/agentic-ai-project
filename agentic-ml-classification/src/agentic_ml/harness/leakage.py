"""
Leakage checks. These run automatically after every split and after
every candidate evaluation. A failure here should block promotion to
the leaderboard, not just produce a warning agents can ignore.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class LeakageCheckResult:
    check_name: str
    passed: bool
    detail: str

    def to_dict(self) -> dict:
        return {"check": self.check_name, "passed": self.passed, "detail": self.detail}


def check_duplicate_rows_across_splits(
    df: pd.DataFrame, train_idx: list[int], val_idx: list[int], test_idx: list[int]
) -> LeakageCheckResult:
    """Exact-duplicate rows appearing in both train and test/val is a classic
    leakage source (e.g. the same customer record entered twice)."""
    train_rows = df.iloc[train_idx].drop_duplicates()
    test_rows = df.iloc[test_idx]
    val_rows = df.iloc[val_idx]

    dup_with_test = pd.merge(train_rows, test_rows, how="inner")
    dup_with_val = pd.merge(train_rows, val_rows, how="inner")

    n_dup = len(dup_with_test) + len(dup_with_val)
    passed = n_dup == 0
    detail = (
        "no exact duplicate rows across splits"
        if passed
        else f"{len(dup_with_test)} train/test dupes, {len(dup_with_val)} train/val dupes"
    )
    return LeakageCheckResult("duplicate_rows_across_splits", passed, detail)


def check_group_overlap(
    df: pd.DataFrame, group_column: str | None, train_idx: list[int], val_idx: list[int], test_idx: list[int]
) -> LeakageCheckResult:
    if not group_column:
        return LeakageCheckResult("group_overlap", True, "no group_column declared, check skipped")

    train_g = set(df[group_column].iloc[train_idx])
    val_g = set(df[group_column].iloc[val_idx])
    test_g = set(df[group_column].iloc[test_idx])

    overlaps = {
        "train_val": train_g & val_g,
        "train_test": train_g & test_g,
        "val_test": val_g & test_g,
    }
    n_overlap = sum(len(v) for v in overlaps.values())
    passed = n_overlap == 0
    detail = "no group overlap between splits" if passed else f"overlaps: { {k: len(v) for k, v in overlaps.items()} }"
    return LeakageCheckResult("group_overlap", passed, detail)


def check_time_ordering(
    df: pd.DataFrame, time_column: str | None, train_idx: list[int], val_idx: list[int], test_idx: list[int],
    strategy: str = "time",
) -> LeakageCheckResult:
    if not time_column:
        return LeakageCheckResult("time_ordering", True, "no time_column declared, check skipped")

    if strategy != "time":
        # Strict global train<val<test ordering is a promise ONLY the pure
        # 'time' strategy makes — checking it against any other strategy
        # fails the split for violating a guarantee it never gave.
        # group_time: each group is isolated to one split (checked by
        # check_group_overlap) and assigned by its earliest timestamp;
        # cross-group calendar overlap is expected and fine. group/random/
        # stratified: a declared time_column just means "exclude this
        # column from features", not "order the folds by it" — a group
        # split with a declared time column previously failed here
        # unconditionally, dead-ending the whole run.
        return LeakageCheckResult(
            "time_ordering", True,
            f"strategy={strategy} makes no global time-ordering promise; "
            "strict train<val<test ordering only applies to strategy=time"
        )

    train_max = df[time_column].iloc[train_idx].max()
    val_min = df[time_column].iloc[val_idx].min()
    val_max = df[time_column].iloc[val_idx].max()
    test_min = df[time_column].iloc[test_idx].min()

    passed = bool(train_max <= val_min) and bool(val_max <= test_min)
    detail = (
        "train ends before val starts, val ends before test starts"
        if passed
        else f"train_max={train_max}, val_range=({val_min},{val_max}), test_min={test_min}"
    )
    return LeakageCheckResult("time_ordering", passed, detail)


def check_fold_class_presence(
    y: pd.Series, train_idx: list[int], val_idx: list[int], test_idx: list[int],
) -> LeakageCheckResult:
    """A fold containing only one target class makes AUC-style metrics
    undefined (sklearn returns NaN with a warning, not an error), which
    then silently poisons every downstream metric and gate that assumes
    both classes are present — including the label_permutation_test gate,
    which compares a NaN mean to a threshold and always fails without
    ever explaining why. Most commonly hit under group/group_time
    strategies when group-to-split assignment happens to correlate with
    the target (e.g. groups whose earliest timestamp is latest are
    disproportionately the ones holding only the "after" label)."""
    folds = {"train": train_idx, "val": val_idx, "test": test_idx}
    degenerate = {
        name: sorted(y.iloc[idx].unique().tolist())
        for name, idx in folds.items() if y.iloc[idx].nunique() < 2
    }
    passed = len(degenerate) == 0
    detail = (
        "every split contains at least 2 target classes"
        if passed
        else f"split(s) with only a single target class present: {degenerate}"
    )
    return LeakageCheckResult("fold_class_presence", passed, detail)


def check_suspicious_feature_correlation(
    X: pd.DataFrame, y: pd.Series, threshold: float = 0.98,
) -> LeakageCheckResult:
    """
    Catches the simple, common case that label_permutation_test does NOT
    catch: a raw feature that is (near-)identical to the target itself
    (e.g. an accidentally-included post-outcome column, or the target
    leaked through an ID join). label_permutation_test instead catches
    pipeline-level leakage (e.g. preprocessing stats computed on the
    full dataset before splitting) — the two checks are complementary,
    not redundant.
    """
    y_numeric = pd.to_numeric(y, errors="coerce")
    suspicious = []
    for col in X.columns:
        col_numeric = pd.to_numeric(X[col], errors="coerce")
        if col_numeric.isna().all():
            continue  # non-numeric column, skip correlation check
        try:
            corr = col_numeric.corr(y_numeric)
        except Exception:
            continue
        if corr is not None and abs(corr) >= threshold:
            suspicious.append((col, round(float(corr), 4)))

    passed = len(suspicious) == 0
    detail = (
        "no feature suspiciously correlated with target"
        if passed
        else f"suspiciously high correlation with target: {suspicious}"
    )
    return LeakageCheckResult("suspicious_feature_correlation", passed, detail)


def label_permutation_test(
    fit_and_score_fn, X_train, y_train, X_val, y_val, metric_name: str,
    n_permutations: int = 5, seed: int = 42, chance_tolerance: float = 0.08,
) -> LeakageCheckResult:
    """
    Fit the candidate PIPELINE (preprocessing + model) on *shuffled*
    training labels. If it still scores well above chance, the pipeline
    itself is leaking information — most commonly because a preprocessing
    step (e.g. a target encoder, or a scaler) was fit on data outside its
    proper scope (the full dataset, or the validation fold).

    This does NOT catch a raw feature that simply equals the target —
    use check_suspicious_feature_correlation for that. The two checks
    are complementary: this one catches leakage through the modeling
    *process*, the correlation check catches leakage through raw
    *feature content*.

    fit_and_score_fn(X_train, y_train, X_val, y_val) -> float (metric value)
    """
    rng = np.random.RandomState(seed)
    y_train_arr = np.asarray(y_train)
    shuffled_scores = []
    for _ in range(n_permutations):
        y_shuffled = rng.permutation(y_train_arr)
        try:
            score = fit_and_score_fn(X_train, y_shuffled, X_val, y_val)
        except Exception as e:
            return LeakageCheckResult(
                "label_permutation_test", False, f"candidate failed to fit on shuffled labels: {e}"
            )
        shuffled_scores.append(score)

    mean_shuffled = float(np.mean(shuffled_scores))
    # for roc_auc / accuracy-like metrics, chance is ~0.5
    chance_baseline = 0.5
    passed = mean_shuffled <= chance_baseline + chance_tolerance
    detail = (
        f"mean {metric_name} on shuffled labels = {mean_shuffled:.4f} "
        f"(chance ~= {chance_baseline}, tolerance = {chance_tolerance})"
    )
    return LeakageCheckResult("label_permutation_test", passed, detail)


def run_all_split_leakage_checks(
    df: pd.DataFrame,
    target_column: str,
    group_column: str | None,
    time_column: str | None,
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
    strategy: str = "time",
) -> list[LeakageCheckResult]:
    return [
        check_duplicate_rows_across_splits(df, train_idx, val_idx, test_idx),
        check_group_overlap(df, group_column, train_idx, val_idx, test_idx),
        check_time_ordering(df, time_column, train_idx, val_idx, test_idx, strategy=strategy),
        check_fold_class_presence(df[target_column], train_idx, val_idx, test_idx),
    ]
