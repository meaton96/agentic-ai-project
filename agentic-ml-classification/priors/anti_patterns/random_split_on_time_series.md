# Anti-Pattern: Random Split on Time-Series / Grouped-Time Data

**What it looks like:** a dataset has a `time_column` (or an implicit
sequence), but the split strategy used is `random` or `stratified`.

**Why it's wrong:** the model can be trained on rows that occur *after*
rows it's being evaluated on. Validation/test metrics will look
artificially strong because the model effectively "saw the future."

**How the harness prevents it:** `make_split()` requires an explicit
`strategy` argument — there is no silent default to `random` when a
`time_column` is declared in the `DatasetSpec`. `check_time_ordering`
enforces strict train→val→test chronological ordering whenever
`strategy="time"`.

**Related but distinct case — `group_time`:** when data has both a
group (e.g. customer) and a time column, `group_time` assigns whole
groups to splits based on each group's earliest timestamp. This
correctly isolates each group to one split while *allowing* different
groups' calendar time ranges to overlap — that overlap is expected and
is not leakage (see `check_time_ordering`'s `strategy="group_time"`
branch, which explicitly does not enforce global ordering).
