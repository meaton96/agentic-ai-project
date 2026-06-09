# `Scheduling/Validation/compare_results.py`
**Language:** Python | **Lines:** 355 | **Last Scanned:** (not provided)

## Purpose
This Python script cross-validates C# DES output against the job_shop_lib Python reference. It compares makespan CSVs and full operation schedules produced by the C# ValidationExporter against the job_shop_lib reference generator. The script is designed to be robust against common data quality issues and provides a detailed report on the comparison results.

## Dependencies
* `pandas` (as `pd`): for data manipulation and analysis
* `json`: for loading schedule JSON files
* `pathlib`: for path manipulation and file existence checks

## Functions
| Function | Signature | Description |
|----------|-----------|-------------|
| `load_data` | `() -> (ref, csharp, ref_schedules, cs_schedules)` | Loads and deduplicates makespan CSVs and schedule JSON files |
| `find_first_divergence` | `(ref_ops, cs_ops) -> list` | Identifies operations where the reference and C# schedules disagree |
| `main` | `() -> int` | Runs the full cross-validation pipeline and prints a report |

## Notes
The script uses a combination of data manipulation and analysis techniques to compare the C# output with the Python reference. It provides a detailed report on the comparison results, including coverage, match/mismatch counts, per-rule summaries, and instance × rule tables. The script also saves the full merged DataFrame to a CSV file for further analysis.