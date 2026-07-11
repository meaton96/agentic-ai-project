# Leakage Rules

These are enforced by `src/agentic_ml/harness/leakage.py`, not just
documented here — this file explains *why*, the code is the actual gate.

## Rule 1: The harness owns the split, always
Agents never see raw file paths, never compute their own train/test
split, and never see the locked test set until the harness explicitly
releases final metrics for the one selected candidate.

## Rule 2: Group-aware splitting for entity data
If rows can be grouped by an entity that could plausibly appear more
than once (customer, machine, patient, session), use `group` or
`group_time` splitting. Random splitting on entity data leaks: the
model can effectively memorize an entity's own history and "predict"
its own outcome via correlated features.

## Rule 3: Time-aware splitting for anything with a timestamp
Random splitting on time-series data lets the model see the future to
predict the past. Use `time` (single entity stream) or `group_time`
(many entities, each with their own timeline).

## Rule 4: Two independent leakage checks, not one
- `label_permutation_test` catches pipeline-level leakage (e.g. a
  preprocessing statistic computed on the full dataset before
  splitting). It fits on shuffled labels and expects chance-level
  performance.
- `check_suspicious_feature_correlation` catches raw feature-level
  leakage (a column that's a near-copy of the target). It does NOT
  require fitting anything — pure correlation check.
These do not overlap. Both must run.

## Rule 5: Preprocessing fit only on train
Any preprocessing step that learns statistics from data (scalers,
imputers, target encoders) must be fit only on the training fold and
applied (not re-fit) to validation/test. sklearn `Pipeline` +
`ColumnTransformer` enforce this automatically as long as `.fit()` is
only ever called with training data — see `baseline_ladder.py`.

## Rule 6: Test set is touched exactly once
Validation/CV metrics drive all iteration and model selection. The
test set is evaluated only for the single final selected candidate,
and that evaluation is not used to pick between candidates.
