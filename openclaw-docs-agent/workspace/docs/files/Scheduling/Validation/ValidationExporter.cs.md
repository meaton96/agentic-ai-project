# `Scheduling/Validation/ValidationExporter.cs`
**Language:** C# | **Lines:** 197 | **Last Scanned:** (not provided)

## Purpose
This file contains the `ValidationExporter` class, which exports makespan results and full operation schedules in a format compatible with the Python reference generator. It enables direct comparison with job_shop_lib output. The class is used to validate the scheduling algorithm by running simulations and exporting the results to CSV and JSON files.

## Dependencies
* `Newtonsoft.Json`: used for JSON serialization and deserialization.
* `Assets.Scripts.Scheduling.Data`: provides data structures for scheduling instances.
* `Assets.Scripts.Scheduling.Core`: provides the core scheduling algorithm and simulator.

## Classes
* `ValidationExporter`: a static class responsible for exporting validation results.
	+ Inheritance: none
	| Method | Signature | Description |
	|--------|-----------|-------------|
	| `readonly` | `(DispatchingRule rule, string key, string fullName)` | Mapping from `DispatchingRule` enum values to short keys and full names. |
	| `RunAndExport` | `(string instanceDir, string outputDir)` | Simulates all Taillard instances in a directory under every mapped rule and writes the results to CSV and JSON export files. |

## Notes
The `ValidationExporter` class only exports results for the five rules present in the `RuleMap`, which matches the rules implemented in the Python reference generator. This ensures that the comparison columns are aligned. The class uses a dictionary to store the schedules for each instance and rule combination, and writes the results to two output files: `csharp_makespans.csv` and `csharp_schedules.json`.