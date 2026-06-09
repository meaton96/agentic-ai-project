# `Scheduling/Validation/generate_reference.py`
**Language:** Python | **Lines:** 175 | **Last Scanned:** Not specified

## Purpose
This Python script generates ground-truth makespans and operation schedules from the job_shop_lib library for cross-validation against a C# DES simulator. It runs every rule in the defined list against every instance in the specified list, then writes two output files: `reference_makespans.csv` and `reference_schedules.json`. The script plays a crucial role in validating the accuracy of the C# simulator by providing a reference point for comparison.

## Dependencies
* `job_shop_lib.benchmarking`: for loading benchmark instances
* `job_shop_lib.dispatching.rules`: for dispatching rule solvers
* `pandas`: for data manipulation and CSV output

## Notes
The script uses a predefined list of instances (`INSTANCES`) and rules (`RULES`) to generate the reference data. The `RULE_MAP` dictionary maps full rule names to short keys used in the output files. The script also extracts the optimum makespan from instance metadata, if available, to calculate the gap percentage. The output files are structured to match the C# simulator's output, allowing for direct comparison and validation.