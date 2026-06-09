# `Scheduling/Validation/CrossValidator.cs`
**Language:** C# | **Lines:** 74 | **Last Scanned:** 2026-06-04

## Purpose
This file contains the `CrossValidator` class, which serves as a Unity entry point for exporting C# makespan results for cross-validation against the job_shop_lib. It simulates every instance under every dispatching rule and writes the results to CSV and JSON files. The exported files are then compared to the Python job_shop_lib reference using a separate script.

## Dependencies
* `UnityEngine`: provides the Unity engine functionality, including the `MonoBehaviour` class that `CrossValidator` inherits from.
* `System.IO`: enables file system operations, such as reading and writing files.

## Classes
The `CrossValidator` class is a Unity `MonoBehaviour` that exports C# makespan results for cross-validation. It inherits from `MonoBehaviour`.
| Method | Signature | Description |
|--------|-----------|-------------|
| `Start` | `void Start()` | Runs the full export pipeline on scene start, including creating the output directory, validating the instance directory, and delegating to `ValidationExporter.RunAndExport`. |

## Notes
The `CrossValidator` component does not use the Unity Resources system, and both the instance directory and output directory must be accessible as real filesystem paths at runtime. The `Start` method logs important information, including the resolved paths for both directories and the command to invoke the comparison script.