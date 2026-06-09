# `Simulation/Jobs/JobData.cs`
**Language:** C# | **Lines:** 136 | **Last Scanned:** 2026-06-04

## Purpose
This file contains the `JobData` class, which encapsulates all persistent data, state, and tracking metrics for a single job in a factory simulation. It plays a crucial role in managing job lifecycle stages, operation sequences, and timing statistics. The class serves as a pure data container, with logic for state transitions and routing handled by a central orchestrator.

## Dependencies
* `Assets.Scripts.Simulation.Machines`: Provides machine-related functionality and types.
* `UnityEngine`: Utilized for visual representation of jobs on the factory floor.

## Classes
### JobData
Role: Encapsulates job data and state. Inheritance: None.
| Method | Signature | Description |
|--------|-----------|-------------|
| GetProcessingTime | `float GetProcessingTime(int machineId)` | Retrieves the processing time for a specific machine on the current operation. |

## Notes
The `JobData` class uses an enum `JobState` to define distinct lifecycle stages of a job, including NeedsRouting, WaitingForPickup, InTransit, Queued, Processing, and Exited. The class also employs properties like `IsLastOperation` and `NextRequiredType` to track job progress and required machine types.