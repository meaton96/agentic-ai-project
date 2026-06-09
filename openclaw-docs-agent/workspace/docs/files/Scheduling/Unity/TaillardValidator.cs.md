# `Scheduling/Unity/TaillardValidator.cs`
**Language:** C# | **Lines:** 157 | **Last Scanned:** 2026-06-04

## Purpose
This file contains the `TaillardValidator` class, which serves as the Unity entry point for loading and validating Taillard JSP benchmark instances. It loads a Taillard instance, runs constraint checks, executes dispatching rules, and optionally prints a Gantt schedule to the console. The class plays a crucial role in validating and testing scheduling algorithms within the Unity environment.

## Dependencies
* `UnityEngine`: for Unity-specific functionality and integration
* `Newtonsoft.Json`: for JSON deserialization of Taillard instances
* `Assets.Scripts.Scheduling.Data`: for Taillard instance data structures
* `Assets.Scripts.Scheduling.Core`: for scheduling core functionality
* `Assets.Scripts.Scheduling.Validation`: for validation and testing utilities

## Classes
The `TaillardValidator` class is a `MonoBehaviour` that loads and validates Taillard instances. It inherits from `MonoBehaviour` to integrate with the Unity environment.
| Method | Signature | Description |
|--------|-----------|-------------|
| LoadAndValidate | `void LoadAndValidate()` | Loads the configured Taillard instance and runs the full validation pipeline |
| LoadInstance | `TaillardInstance LoadInstance(string fileName)` | Loads and deserializes a Taillard JSON instance from the Unity Resources system |

## Notes
The `TaillardValidator` class uses a combination of Unity's `Resources` system and `Newtonsoft.Json` for JSON deserialization to load Taillard instances. The `LoadAndValidate` method executes a series of steps, including constraint validation, dispatching rule execution, and optional Gantt schedule printing. The class also provides a context menu entry for manual validation without entering Play Mode.