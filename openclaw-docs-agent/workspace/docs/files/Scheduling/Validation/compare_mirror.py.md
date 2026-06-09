# `Scheduling/Validation/compare_mirror.py`
**Language:** Python | **Lines:** 194 | **Last Scanned:** 2026-06-04

## Purpose
This script compares the makespan results of C# and Python DES implementations to identify any discrepancies. It loads the results from CSV files, merges them, and computes a match summary. The script also diagnoses mismatches by finding the first divergent operation between the two implementations.

## Dependencies
* `pandas` (as `pd`): for data manipulation and analysis
* `json`: for loading schedule data from JSON files
* `pathlib`: for working with file paths

## Functions
| Function | Signature | Description |
|----------|-----------|-------------|
| `main` | `()` | Entry point: loads, merges, and compares C# and mirror DES CSV results |

## Notes
The script assumes that the C# and Python DES implementations share the same architecture, with per-machine queues and identical dispatching rules. Any discrepancies in the output indicate a bug in one of the implementations. The script also relies on the presence of specific input files (`csharp_makespans.csv`, `mirror_des_makespans.csv`, `csharp_schedules.json`, and `mirror_des_schedules.json`) in the working directory.