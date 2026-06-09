# `Simulation/Types/ConfigLoader.cs`
**Language:** C# | **Lines:** 550 | **Last Scanned:** 2026-06-04

## Purpose
This file contains the `ConfigLoader` class, which is responsible for loading FJSSPConfig instances from JSON files on disk. It supports both single-config files and batch arrays for headless runs, mapping string-based machine type names to `MachineType` enum values at load time. The loaded configurations are used to initialize simulations.

## Dependencies
* `System.IO`: for file input/output operations
* `UnityEngine`: for Unity-specific functionality
* `Assets.Scripts.Simulation.Machines`: for machine-related data structures and logic
* `Assets.Scripts.Simulation.Logging`: for logging and error handling

## Classes
### ConfigLoader
Role: Loads FJSSPConfig instances from JSON files.
Inheritance: None (static class).
| Method | Signature | Description |
|--------|-----------|-------------|
| LoadSingle | `FJSSPConfig LoadSingle(string path)` | Loads a single FJSSPConfig from a JSON file path. |
| LoadBatch | `FJSSPConfig[] LoadBatch(string path)` | Loads an array of FJSSPConfig instances from a batch JSON file. |
| ParseSingle | `FJSSPConfig ParseSingle(string json)` | Parses a single FJSSPConfig from a JSON string. |
| ParseBatch | `FJSSPConfig[] ParseBatch(string json)` | Parses a batch of FJSSPConfig instances from a JSON string. |
| Convert | `FJSSPConfig Convert(JsonConfig raw)` | Converts a `JsonConfig` to a runtime FJSSPConfig. |
| ParseMachineType | `MachineType? ParseMachineType(string name)` | Parses a machine type name string to its corresponding `MachineType` enum value. |

### JsonBatchWrapper
Role: Wrapper for batch config JSON with a "configs" array field.
Inheritance: None.
No methods.

### JsonConfig
Role: Serializable representation of a single FJSSPConfig for JSON deserialization.
Inheritance: None.
No methods.

### JsonStochasticConfig
Role: Serializable representation of stochastic configuration data.
Inheritance: None.
No methods.

## Notes
The `ConfigLoader` class uses Unity's `JsonUtility` for JSON deserialization, which requires the use of `[Serializable]` attributes on the data classes (`JsonBatchWrapper`, `JsonConfig`, and `JsonStochasticConfig`). The `Convert` method performs the actual conversion from `JsonConfig` to `FJSSPConfig`, including the mapping of machine type names to `MachineType` enum values.