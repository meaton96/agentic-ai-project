# Baseline Ladder

Always run these in order before any agent-generated candidate is
trusted. A candidate that can't beat `logistic_regression` needs a good
explanation before it's worth the extra complexity.

1. `dummy_most_frequent` — floor. If a "sophisticated" candidate doesn't
   clear this by a wide margin, something is wrong with either the
   candidate or the target definition.
2. `logistic_regression` — cheap, interpretable, tells you if the
   problem is roughly linearly separable after basic preprocessing.
3. `random_forest` — cheap non-linear baseline, robust to messy mixed
   data, minimal tuning required.
4. `hist_gradient_boosting` — sklearn-native gradient boosting, no
   extra dependency, usually close to LightGBM/XGBoost quality.
5. `lightgbm` / `xgboost` — typically the strongest tabular baselines;
   worth the dependency for most real datasets.

An agent proposing a "novel" architecture should be evaluated against
all five of these first. If it doesn't clear `hist_gradient_boosting`,
the extra complexity likely isn't earning its keep for this dataset.
