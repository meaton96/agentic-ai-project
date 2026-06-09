# `Simulation/EpisodeTracker.cs`
**Language:** C# | **Lines:** 550 | **Last Scanned:** 2026-06-04

## Purpose
The `EpisodeTracker` class accumulates per-episode and per-machine statistics during a simulation run, managing all statistical tracking including processing times, failure counts, repair durations, and machine availability metrics. It plays a crucial role in the project by providing insights into the simulation's performance and behavior. This class is owned as a field by `SimulationBridge` and is responsible for constructing the final `EpisodeRecord` at the end of each episode.

## Dependencies
* `System.Collections.Generic`: for dictionary and enumerable operations
* `UnityEngine`: for simulation-related functionality
* `Assets.Scripts.Simulation.Machines`: for machine-related data and operations
* `Assets.Scripts.Simulation.Types`: for simulation-related data types
* `Assets.Scripts.Simulation.Logging`: for logging and debugging purposes

## Classes
### EpisodeTracker
Role: Accumulates per-episode and per-machine statistics during a simulation run.
Inheritance: None
| Method | Signature | Description |
|--------|-----------|-------------|
| Reset | `void Reset()` | Clears all accumulated statistics at the start of every episode. |
| RecordOperationStart | `void RecordOperationStart(int machineId)` | Records that a machine has started an operation. |
| AddProcessingTime | `void AddProcessingTime(int machineId, double dt)` | Accumulates processing time for a machine during a frame where it is active. |
| RecordOperationComplete | `void RecordOperationComplete(int machineId)` | Records that an operation has completed on a machine. |
| RecordMachineFailure | `void RecordMachineFailure(int machineId, float repairDuration, double simTime)` | Records a machine failure event. |
| RecordRepairComplete | `void RecordRepairComplete(int machineId, double simTime)` | Records that a machine has returned to operational state after repair. |
| Build | `EpisodeRecord Build(FJSSPConfig config, double simTime, string ruleName, int completedJobs, int totalOps, int decisionPoints, double totalReward, int agvCount, IEnumerable<PhysicalMachine> machines, float averageTimeScale = 100f)` | Constructs the final `EpisodeRecord` from all accumulated statistics. |
| TheoreticalMeanTTF | `static float TheoreticalMeanTTF(float lambda)` | Computes the theoretical mean time-to-failure (TTF) for a Weibull distribution. |

## Notes
The `EpisodeTracker` class uses dictionaries to store per-machine statistics, allowing for efficient lookup and accumulation of data. The `Build` method constructs the final `EpisodeRecord` by aggregating data from these dictionaries and other episode-level statistics. The `TheoreticalMeanTTF` method provides a utility function for calculating the mean time-to-failure for a Weibull distribution, which can be useful in analyzing machine failure patterns.