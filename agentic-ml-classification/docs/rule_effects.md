# Harness Rules and Their Measured Effects

*One view combining [`harness_pseudocode.md`](harness_pseudocode.md) (what
each rule checks) with [`ablation_study_report.md`](ablation_study_report.md)
(what actually happens when a rule is disabled) — for each step, the rule
immediately followed by its measured effect, not two documents to cross-
reference by hand. Where a rule hasn't been run through the ablation
harness yet, that's stated plainly rather than implied — this doc tracks
coverage, it doesn't pretend it's complete.*

## Coverage at a glance

| Step | Rules | Ablation-tested? | Headline finding |
|---|---|---|---|
| Intake | 6 | ✅ all 6 | severities range from *no-op* to *train fold ends up completely empty* |
| Feature Engineering | 5 | ✅ all 5 | 3 distinct failure shapes: crash, silent garbage, deferred crash |
| Split Resolution | 4 | ✅ all 4 | 3 checks degrade *which* exception you get, not safety; 1 check (reconciliation) costs nothing to disable |
| Split-Level Leakage Checks | 4 | ✅ all 4 | 3 checks have zero cross-check redundancy; 1 backstopped only by an emergent NaN-comparison side effect |
| Modeling Candidate (structural) | 5 | ✅ all 5 | first fixture to test adversarial-code containment, not statistical leakage; 1 more asymmetric gap found |
| Candidate Leakage Gates | 3 | ✅ all 3 | 2 gates each independently necessary; 1 gap found and closed |
| Verification | 1 | ✅ | a **real, previously-shipped gap** — not a hypothetical ablation — found and fixed |
| Finalize | 0* | ✅ | *not an independent rule — fully absorbed into Dynamic Planner's precondition check |
| Dynamic Planner | 2 | ✅ all 2 | one check enforces the Finalize guarantee, the Verification backstop's registry layer, and every other precondition in the system |

**30 of 30 rules — full coverage.** The study closes the same way
several of its phases opened: by discovering the inventory's assumed
count was wrong. "Finalize" was never an independently-ablatable rule —
`steps/finalize_step.py` has no check of its own; the one-shot guarantee
is entirely Dynamic Planner's `required_state` precondition, evaluated
against one specific registry entry. Net rule count for this area: 3
(registry check, precondition check, verification gate-status check),
not the assumed 4. See [`scripts/run_ablation_study.py`](../scripts/run_ablation_study.py)
Phases 1/2 (leakage gates), 1b (feature engineering), 1c (intake), 1d
(modeling-candidate structure), 1e (split resolution), 1f (split-level
leakage checks), and 1g (dynamic planner / verification / finalize) for
all of it.

---

## Intake

```
if target_column not in dataset
    reject
if unique(target_column) < 2 or > MAX_CLASSES
    reject
else
    task ← binary if 2 classes else multiclass     (agent's task field unused)
if group_column or time_column not in dataset
    reject
if group_column or time_column == target_column
    reject
if id_columns is not a list
    reject
if any id_column not in dataset, or == target_column
    reject
```

**Measured effect — all 6 checks tested** (`docs/ablation_study_report.md`
Phase 1c). This step turned out to be the most varied in the whole study
— severity ranges from *catastrophic* to *genuinely harmless*, not a
uniform "reject or bad things happen":

| Rule disabled | What actually happens |
|---|---|
| target existence | **crashes inside the validation function itself** — `df[target]` two lines past where the check used to short-circuit raises `KeyError` |
| cardinality, lower bound (n=1) | accepted at intake, but an unrelated split-level check (`check_fold_class_presence`) independently catches it — genuine defense in depth |
| cardinality, upper bound (n=30) | accepted, split succeeds, no crash — but downstream multiclass metric computation grows measurably more fragile to class/fold mismatches as cardinality rises relative to fold size (this is the practical reason `MAX_CLASSES=20` exists as a margin, not an arbitrary cutoff) |
| group/time existence | **crashes** — `KeyError` inside `make_split` |
| group/time == target_column | **the most severe finding in this entire study**: accepted, then group-based splitting assigns entire classes to entire folds — in the tested fixture, the **training fold ends up with zero rows**. `check_group_overlap` reports nothing wrong (there genuinely is no group overlap); only `check_fold_class_presence` catches it |
| `id_columns` type (must be a list) | a string instead of a list gets unpacked **character-by-character** (Python string iteration); a genuine feature column silently vanishes from the model if its name happens to match one of the characters — no error at any layer |
| `id_columns` existence / target-collision | **near-total no-op** — `LoadedDataset.X`'s own `if c in df.columns` filter already absorbs both faults defensively; disabling this check only loses the clean error message, not correctness |

Two things worth naming explicitly. First, the group/time-vs-target
collision check is the single highest-consequence rule found by this
study so far — worse than any Feature Engineering finding — precisely
because its failure mode isn't a crash a human would notice immediately,
it's a *quietly empty training set* that a less-attentive caller could
miss. Second, not every rule in this inventory is equally load-bearing:
the `id_columns` existence/collision check turned out to be almost pure
redundancy with a defensive filter one layer down, which is itself a
useful, honest finding — this study isn't only good for finding gaps, it
also tells you which checks are doing real work and which ones aren't.

---

## Feature Engineering

```
if op_id not in FEATURE_OPS
    reject
if column == target_column
    reject
if op is numeric and column dtype is not numeric
    reject
if op == datetime_parts and column is not datetime-like
    reject
if drop_column == group_column or drop_column == time_column
    reject
else
    accept
```

**Measured effect — all 5 rules tested** (`docs/ablation_study_report.md`
Phase 1b):

| Rule disabled | What actually happens |
|---|---|
| `op_id` in catalog | **crashes**: `KeyError: "Unknown feature op 'not_a_real_op'"` |
| input ≠ target_column | applies cleanly — derived column correlates **0.836** with target (below the 0.98 downstream gate threshold — **not** backstopped later) |
| numeric dtype for numeric ops | **crashes**: `TypeError: operation 'truediv' not supported for dtype 'str'` |
| datetime-like for `datetime_parts` | **no crash** — silently produces `1970` for every row |
| group/time column protected from drop | drops cleanly, then breaks one stage later: `make_split(strategy="group")` raises `KeyError` |

None of these five degrade gracefully on their own — the pseudocode's
`reject` is not a formality anywhere in this table, and the
target-column check specifically is not redundant with any downstream
gate (0.836 < 0.98).

---

## Split Resolution

```
if strategy not in STRATEGIES
    error
if strategy needs group_column and none declared
    if a likely group column was detected
        use it
    else
        error
if strategy needs time_column and none declared
    if a likely time column was detected
        use it
    else
        error
```

**Measured effect — all 4 checks tested** (`docs/ablation_study_report.md`
Phase 1e). This is really two functions, not one: `resolve_split_columns`
does the "use it" auto-fill (a recovery mechanism, not a reject gate),
`make_split` does the "error" (three real checks: strategy validity,
group required, time required).

| Rule disabled | What actually happens |
|---|---|
| strategy validity | **crashes**, but with a *different, less informative* exception — falls through every `elif` to the guarded `else: raise AssertionError` at the bottom of `make_split`, losing the clear "Unknown split strategy" message |
| group_column required | **crashes** with `KeyError: None` (from `df[None].values`) — gives no hint the real problem is a missing `group_column` |
| time_column required | same — `KeyError: None`, same loss of diagnostic clarity |
| split-column reconciliation (auto-fill) | **no safety cost at all** — `make_split`'s own required-column check still catches the unresolved `None` cleanly; only the auto-fill convenience is lost |

**The one rule in this whole study whose removal costs nothing.** Every
other ablated rule so far degrades safety, diagnosability, or both when
disabled. `resolve_split_columns`'s reconciliation is different in kind
— it's a UX/convenience layer sitting entirely in front of a real safety
check that stays intact regardless. The other three are the opposite
pattern from most of this study: none of them produce a *silent*
failure (every one still crashes), but all three degrade *which*
exception a caller sees — an `AssertionError` with no message, or a
`KeyError: None` that doesn't name the actual missing column. A caller
that specifically catches `ValueError` (as `make_split`'s own docstring
implies is the contract) would not catch either ablated failure mode.

---

## Split-Level Leakage Checks

```
fail if duplicate rows appear across splits
fail if a group id appears in more than one split
fail if time ordering is violated
fail if a class is missing from any fold
else pass                              (all four run, independently)
```

Corrected from the original five-line version of this pseudocode: there
is no split-level correlation check.
`check_suspicious_feature_correlation` is only ever called from
`modeling_step.py`, scoped to a specific candidate's selected columns —
already covered under Candidate Leakage Gates below.

**Measured effect — all 4 checks tested** (`docs/ablation_study_report.md`
Phase 1f):

| Rule disabled | What actually happens |
|---|---|
| duplicate rows across splits | **accepted, zero cross-check redundancy** — an exact-duplicate row in both train and val slips through with nothing else catching it |
| group overlap | **accepted, zero cross-check redundancy** — same pattern, verified with a fixture free of accidental exact-row duplication |
| time ordering | **accepted, zero cross-check redundancy** — a `"time"`-strategy split with train given the *latest* dates and val the *earliest* passes cleanly |
| fold class presence | **accepted at the split level, but backstopped downstream — by accident, not by design** |

**The fold-class-presence finding is the most interesting one in this
phase.** A single-class validation fold reaching the modeling step
produces `score = NaN` (`sklearn`'s `roc_auc_score` is undefined with
one class present). The downstream permutation gate still rejects the
resulting candidate — but only because `NaN <= threshold` evaluates to
`False` in Python/NumPy regardless of `threshold`, not because anything
was written to detect "did I just receive a NaN metric." Verified
directly: `label_permutation_test` on this fixture returns
`passed=False` with `mean roc_auc on shuffled labels = nan` in its own
detail string. This is a real backstop, but a fragile one — it depends
on NaN-propagation happening to reach a `<=` comparison rather than,
say, being coerced to `0` or raising its own exception somewhere in the
metric pipeline first.

**The other three checks have no such luck.** Duplicate rows, group
overlap, and time ordering are each the *sole* mechanism in the entire
system that would catch their specific fault — verified by checking
whether any of the other three checks in the same
`run_all_split_leakage_checks` call also failed (they didn't, in any of
the three ablated cases). Combined with Phase 1e's finding that
`resolve_split_columns`' reconciliation is the one rule that costs
nothing to disable, this phase reinforces the emerging pattern: most
rules in this system are the *only* thing standing between a proposal
and a real correctness failure, not one layer of several.

---

## Modeling Candidate (structural checks)

```
if candidate shape is malformed
    reject
if a referenced column is hallucinated, or is target/id/group/time
    reject
if template_id not in TEMPLATES
    reject
if config is missing a required key
    reject
if source imports FORBIDDEN_IMPORTS or calls FORBIDDEN_CALLS
    reject
if sandbox execution times out or raises
    reject
else
    accept unfitted pipeline           (only object allowed to cross back)
```

**Measured effect — all 5 checks tested** (`docs/ablation_study_report.md`
Phase 1d; the pseudocode's "template_id" and "config" lines share one
`AblationConfig` flag since both raise through the same code path).
This is the first phase in the study to test containment of *actively
adversarial* code, not just statistically leaky candidates:

| Rule disabled | What actually happens |
|---|---|
| candidate shape | **crashes**: `KeyError: 'template_id'` |
| column check — target column selected as a feature | **still caught downstream** by the feature-correlation leakage gate (raw correlation = 1.0) |
| column check — likely-ID column selected as a feature | **accepted with no errors** — unlike the target-column case, an uncorrelated ID column has nothing for any leakage gate to catch |
| template/config — unknown `template_id` | **crashes**: `KeyError` from `get_template` |
| template/config — missing required config key | **backstopped** — the sandbox catches the resulting `KeyError` from inside `build_pipeline()` and converts it to a normal rejection, not a crash |
| static AST check | a template importing `os` **genuinely executes** inside the sandboxed subprocess (verified: it correctly reports its own tempdir cwd) — contained to that subprocess, but the forbidden import itself is no longer blocked |
| sandbox build-error check | **crashes**: `AttributeError: 'NoneType' object has no attribute 'fit'` when a build failure is silently ignored and `pipeline=None` reaches `.fit()` |

**The column check reproduces Intake's asymmetry exactly.** A direct,
raw-value leak (the target column itself) gets backstopped by an
independent downstream gate; an ID-like column with no real statistical
relationship to the target does not — nothing anywhere in the pipeline
is positioned to catch it once this one check is gone. Same shape of
finding as the group/time-vs-target case in Intake, in a completely
different step.

**The AST check finding is qualitatively different from everything else
in this study.** Disabling it doesn't produce a leaked metric or a
crash from bad data — it produces a template that runs `import os` and
proves it, safely, inside its own sandboxed subprocess (the fault
fixture only reads its own tempdir path; it never touches anything
outside the subprocess). Static code containment and statistical
leakage detection are different problems solved by different mechanisms
in this harness, and this is the first fixture in the study to actually
exercise the first one.

---

## Candidate Leakage Gates

```
fail if mean(score on PERM_N label-shuffles) > 0.5 + TOLERANCE
fail if any |corr(selected_column, target)| >= CORR_THRESHOLD
fail if (train_score - holdout_score) - (shuffled_train_score - shuffled_holdout_score) > EXCESS_GAP_TOLERANCE
else pass                              (all three required, none is redundant)
```

**Measured effect — all 3 gates tested** (`docs/ablation_study_report.md`
Phase 1 + Phase 2), across three fault scenarios:

| Scenario | correlation gate | permutation gate | train-CV-consistency gate |
|---|---|---|---|
| 1 — raw near-duplicate column | **catches** | misses | misses |
| 2 — encoder ignores its fold (closure over true labels) | misses | misses | misses |
| 3 — pipeline bypasses `fit()`'s y entirely | misses | **catches** | misses |

Gates 1 and 2 (correlation, permutation) are each independently
necessary for the scenario they catch — disabling either one, alone,
flips exactly the scenario it was built for from rejected to accepted.
Scenario 2 clears **all three** gates at every setting, including
production defaults — a real, reproducible gap. Closing it wasn't
free: the first version of the third gate (an absolute train-vs-holdout
score gap) correctly caught the synthetic leak but **broke 4 real
tests**, rejecting a legitimate gradient-boosted-tree candidate whose
ordinary model-capacity overfitting (gap 0.54) was larger than any
threshold that would still catch a genuine leak. The shipped version
compares against a shuffled-label baseline instead (excess gap
tolerance 0.15) — verified against both the exact case that broke
(excess gap 0.11, passes) and a deliberately extreme non-leaky control
(unregularized decision tree memorizing noise — excess gap 0.05,
passes).

Follow-up also found the honest limit of gate 3: Scenario 2's specific
leak (a closure bypassing `fit()`'s arguments) defeats *any* refit-based
test, including this new one — but that exact bug class is structurally
impossible for a real template in this system to construct (§Step 5,
layer 5: `build_pipeline(config)` never receives data). A more realistic,
properly-scoped version of the same idea was built and, empirically, did
**not** meaningfully inflate validation performance — the residual risk
is real but narrower than the original finding suggested.

---

## Verification, Finalize, and Dynamic Planner

These three sections of the original inventory turned out to be one
mechanism wearing three names, plus one real bug this study found and
fixed. Covered together because the code does.

```
Dynamic Planner:
    if agent_id not in registry
        reject
    if a required_state precondition is not met
        reject
    else
        accept

Finalize:
    (no check of its own — see below)

Verification:
    if LLM output is unparseable
        verdict ← flagged
    else
        verdict ← approved | flagged | rejected
    verdict can only downgrade a passing candidate, never upgrade a failed one
```

**Finalize has no independent check.** `steps/finalize_step.py` will
happily refit and re-evaluate on the test set if called a second time —
nothing inside it stops that. The entire one-shot guarantee is Dynamic
Planner's `required_state={"final_test_metrics_present": False}`
precondition, evaluated against the `"finalize"` registry entry. "3
sections, 4 assumed rules" collapses to 3 real ones: the registry check,
the precondition check (which *is* the Finalize guard), and Verification's
gate-status check.

**Measured effect — all 3 tested:**

| Rule disabled | What actually happens |
|---|---|
| Dynamic Planner registry check | **crashes inside `validate_plan` itself** — `KeyError` from `get_agent()`, one function, not a downstream surprise |
| Dynamic Planner precondition check | **silently permits a second `finalize` proposal** after the test set was already touched once — the exact mechanism that makes Finalize's guarantee real, gone |
| Verification gate-status check | **a real, previously-shipped gap** — see below |

**The verification finding is the most significant result in the entire
study, and it isn't a hypothetical ablation.** `execute_agent_step`'s
`"verification"` branch selects a candidate via
`args.get("candidate_id") or state.best_unverified_candidate_id()`.
`best_unverified_candidate_id()` correctly filters to `passed_gate=True`
candidates — but if a planner proposal supplies an explicit
`candidate_id` in `args` (naming *any* candidate, gate-failed included —
`ctx.modeling_results` stores every candidate ever proposed, not just
the passing ones), that filter is never consulted. Nothing downstream
re-checked gate status before this study. Confirmed directly: a
fixture with one passing and one gate-failed (leaky) candidate, an
explicit `args={"candidate_id": "<the failed one>"}`, and a stubbed LLM
that always says "approved" — the failed candidate's
`verification_verdict` came back `"approved"`. This directly violates
the project's own stated invariant (`CLAUDE.md` #4: the verification
agent "must never be shown... one that failed").

**Fixed directly**, not just documented: `execute_agent_step` now checks
`candidate.ok` immediately after resolving `candidate_id`, before
`build_review_bundle` or the LLM call ever happen
(`orchestrator/dynamic_loop.py`). `AblationConfig.skip_verification_gate_status_check`
reproduces the original bug on demand, so the finding stays reproducible
in the same before/after shape as everything else in this study rather
than existing only as a one-time discovery.

---

## What this doc doesn't claim

**30 of 30 identified rules have been empirically ablated.** Coverage is
complete for the rule inventory as currently understood — which is a
narrower claim than "every possible bug has been found." Two limits
worth being explicit about:

1. **The inventory itself was revised five separate times** while
   building this (Intake 2→6, Modeling Candidate 6→5, Split Resolution
   2→4, Split-Level Leakage 5→4, Verification/Finalize/Dynamic Planner
   4→3) — each time because reading the real code turned up more or
   fewer distinct checks than the original pseudocode pass assumed. A
   sixth undiscovered miscount elsewhere in the system is not something
   this document can rule out; it can only report that none turned up
   in what was actually read.
2. **"Ablated" means "the specific fault fixture built for this rule was
   tested,"** not "every possible fault this rule could ever catch was
   tested." Phase 1g's verification finding is the clearest illustration
   of the gap between those two claims: the rule *looked* fully specified
   by its pseudocode line, and the gap that mattered was in a code path
   (`args`-based explicit targeting) the pseudocode simply didn't mention.

The methodology that got this far — a fault fixture, an `AblationConfig`
flag, a leave-one-out run, and, as Phases 2 and 1g both showed,
willingness to fix what the results turn up rather than just document
it — is the actual deliverable here, more than the specific number of
rules covered. Applying it to a part of the system not covered by this
inventory at all (the six agents' own prompts, the MCP fact server, the
priors hierarchy) is the natural next extension, not a continuation of
this same list.
