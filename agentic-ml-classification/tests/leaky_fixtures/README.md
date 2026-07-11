# Leaky Fixtures

Small synthetic datasets with known, intentional leakage, used as
regression tests to make sure the leakage checks never silently stop
catching what they're supposed to catch.

- `obvious_feature_leak.csv` — `churn_flag_copy` is a literal copy of
  the target `churned`. `check_suspicious_feature_correlation` must
  flag this every time; if a future change to the harness stops
  catching it, that's a regression.
