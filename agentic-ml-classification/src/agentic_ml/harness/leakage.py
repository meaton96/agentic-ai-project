"""
Leakage checks. These run automatically after every split and after
every candidate evaluation. A failure here should block promotion to
the leaderboard, not just produce a warning agents can ignore.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from agentic_ml.ablation import AblationConfig


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


def _train_holdout_gap(fit_and_score_fn, X_train, y_train, holdout_frac: float, seed: int) -> float:
    """in_sample_score - held_out_score for one label assignment, using a
    SINGLE internal train/holdout split rather than full n-fold CV —
    2 fits instead of (1 + n_folds). K-fold CV was the first version of
    this check (see train_cv_consistency_check's docstring for why it
    had to change) and was accurate but far too expensive: 18 refits per
    candidate made the test suite time out. A single holdout split is
    the same idea (score on data the model wasn't fit on, from within
    the training fold) at a cost comparable to label_permutation_test's
    own 5 refits, which is what actually has to run on every candidate."""
    rng = np.random.RandomState(seed)
    n = len(X_train)
    idx = np.arange(n)
    rng.shuffle(idx)
    split_at = max(1, int(n * (1 - holdout_frac)))
    fit_idx, holdout_idx = idx[:split_at], idx[split_at:]

    X_fit = X_train.iloc[fit_idx] if hasattr(X_train, "iloc") else X_train[fit_idx]
    y_fit = y_train.iloc[fit_idx] if hasattr(y_train, "iloc") else y_train[fit_idx]
    X_ho = X_train.iloc[holdout_idx] if hasattr(X_train, "iloc") else X_train[holdout_idx]
    y_ho = y_train.iloc[holdout_idx] if hasattr(y_train, "iloc") else y_train[holdout_idx]

    in_sample_score = fit_and_score_fn(X_fit, y_fit, X_fit, y_fit)
    holdout_score = fit_and_score_fn(X_fit, y_fit, X_ho, y_ho)
    return in_sample_score - holdout_score


def train_cv_consistency_check(
    fit_and_score_fn, X_train, y_train, metric_name: str,
    holdout_frac: float = 0.3, n_baseline_permutations: int = 1, seed: int = 42, excess_gap_tolerance: float = 0.15,
) -> LeakageCheckResult:
    """
    Added after an ablation study (docs/ablation_study_report.md,
    Scenario 2) found a real gap: label_permutation_test does NOT
    reliably catch a preprocessing component (e.g. a target encoder)
    that is properly scoped to whatever (X, y) it's given via fit(), but
    is internally self-referential — i.e. not cross-fitted, so a
    training row's own label leaks into its own encoded feature value.
    That gap exists because label_permutation_test only ever refits on
    a DIFFERENT (shuffled) y — it never inspects the pipeline's behavior
    on the SAME data it was fit on, which is exactly where in-sample
    self-referential bias shows up.

    Two iterations to get here, both worth recording:
    - v1 compared in-sample score to a 5-fold CV score against a FIXED
      absolute gap threshold. Broke on real candidates using
      high-capacity templates (gradient-boosted trees) on a small stub
      dataset: gap=0.54 from ordinary model capacity, not leakage — an
      absolute threshold can't tell a tree's normal overfitting from a
      genuine leak when the former can be larger.
    - v2 fixed the false-positive by comparing the real train-vs-holdout
      gap to the SAME gap computed under label permutation (capacity-
      driven overfitting shows up similarly whether fitting real or
      shuffled labels; what's left after subtracting is leakage-specific)
      — but used full n-fold CV for both, an 18-refit cost per candidate
      that made the test suite time out. This version keeps v2's
      relative-to-permutation-baseline logic but replaces n-fold CV with
      a single internal train/holdout split (4 refits total: in-sample +
      holdout, for both the real and one shuffled pass) — comparable
      cost to label_permutation_test's own 5 refits, which already runs
      on every candidate.

    This does NOT catch a leak that bypasses the (X, y) arguments to
    fit() entirely (e.g. a component reading a closed-over reference
    instead of its actual arguments) — no refit-based statistical test
    can, since every split and every permutation would be equally
    "poisoned," so the real and baseline gaps would move together and
    the excess would stay near zero. That class of bug is prevented
    structurally in this project by the sandbox contract
    (harness/sandbox.py): a template's build_pipeline(config) never
    receives data, so it cannot construct such a reference in the first
    place. See docs/ablation_study_report.md for both worked examples,
    including one where a real (non-closure) self-referential encoder
    was constructed and, empirically, did NOT meaningfully inflate
    validation-fold performance either — the risk this check guards
    against is real but narrower than it first appears.

    fit_and_score_fn(X_train, y_train, X_val, y_val) -> float (metric value)
    """
    try:
        real_gap = _train_holdout_gap(fit_and_score_fn, X_train, y_train, holdout_frac, seed)
    except Exception as e:
        return LeakageCheckResult("train_cv_consistency", False, f"candidate failed to fit in-sample: {e}")

    rng = np.random.RandomState(seed)
    y_arr = np.asarray(y_train)
    baseline_gaps = []
    for _ in range(n_baseline_permutations):
        y_shuffled = rng.permutation(y_arr)
        try:
            baseline_gaps.append(_train_holdout_gap(fit_and_score_fn, X_train, y_shuffled, holdout_frac, seed))
        except Exception:
            continue

    if not baseline_gaps:
        return LeakageCheckResult("train_cv_consistency", False, "could not compute a shuffled-label baseline gap")

    baseline_gap = float(np.mean(baseline_gaps))
    excess_gap = real_gap - baseline_gap
    passed = excess_gap <= excess_gap_tolerance
    detail = (
        f"real train-vs-holdout gap={real_gap:.4f}; shuffled-label baseline gap={baseline_gap:.4f} "
        f"(excess gap={excess_gap:.4f}, tolerance={excess_gap_tolerance})"
    )
    return LeakageCheckResult("train_cv_consistency", passed, detail)


def run_all_split_leakage_checks(
    df: pd.DataFrame,
    target_column: str,
    group_column: str | None,
    time_column: str | None,
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
    strategy: str = "time",
    ablation: Optional[AblationConfig] = None,
) -> list[LeakageCheckResult]:
    """ablation: research-only, see agentic_ml.ablation — every flag
    defaults to False, so ablation=None is identical to omitting it. A
    skipped check is replaced with a trivially-passing result (not
    omitted from the list) so run_split_step's `all(c.passed ...)` sees
    the same shape of result it always does — this is what "the check
    doesn't run" actually looks like to a caller, not a shorter list."""
    ablation = ablation or AblationConfig()

    def _skip(name: str) -> LeakageCheckResult:
        return LeakageCheckResult(name, True, "SKIPPED (ablation)")

    return [
        _skip("duplicate_rows_across_splits") if ablation.skip_duplicate_rows_check
        else check_duplicate_rows_across_splits(df, train_idx, val_idx, test_idx),
        _skip("group_overlap") if ablation.skip_split_group_overlap_check
        else check_group_overlap(df, group_column, train_idx, val_idx, test_idx),
        _skip("time_ordering") if ablation.skip_time_ordering_check
        else check_time_ordering(df, time_column, train_idx, val_idx, test_idx, strategy=strategy),
        _skip("fold_class_presence") if ablation.skip_split_fold_class_presence_check
        else check_fold_class_presence(df[target_column], train_idx, val_idx, test_idx),
    ]
