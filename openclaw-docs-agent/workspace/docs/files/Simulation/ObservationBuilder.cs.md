# `Simulation/ObservationBuilder.cs`
**Language:** C# | **Lines:** 12000+ | **Last Scanned:** 2026-06-04

## Purpose
This file contains the `ObservationBuilder` class, which is responsible for constructing observation data for machine learning agents in a factory simulation environment. The class builds various observation streams, including spatial occupancy grids, scheduling matrices, and global scalar features. These observations are used to inform decision-making in the simulation.

## Dependencies
* `UnityEngine`: for Unity-specific functionality and data types
* `Assets.Scripts.Simulation.Types`: for simulation-specific data types and constants
* `Assets.Scripts.Simulation.Jobs`: for job-related data and functionality
* `Assets.Scripts.Simulation.Machines`: for machine-related data and functionality
* `Assets.Scripts.Simulation.AGV`: for automated guided vehicle (AGV) related data and functionality
* `Assets.Scripts.Simulation.FactoryLayout`: for factory layout and configuration data

## Classes
### ObservationBuilder
Role: builds observation data for machine learning agents. Inheritance: none.
| Method | Signature | Description |
|--------|-----------|-------------|
| `BuildCompleteSnapshot` | `float[] BuildCompleteSnapshot(DecisionRequest currentDecision)` | Builds the complete observation snapshot sent to ML-Agents |
| `BuildSpatialOccupancyGrid` | `float[] BuildSpatialOccupancyGrid()` | Builds the 64x64x3 spatial occupancy grid |
| `BuildSchedulingMatrix` | `float[] BuildSchedulingMatrix()` | Builds the scheduling matrix of shape (MaxJobs, 2 * MaxMachines, 3) |
| `BuildGlobalScalars` | `float[] BuildGlobalScalars()` | Builds the 10-dimensional global scalar feature vector |
| `BuildDistanceMatrix` | `float[] BuildDistanceMatrix()` | Builds the distance matrix |
| `BuildEventFlags` | `float[] BuildEventFlags(DecisionRequest req)` | Builds the event flags |
| `ApplyDomainRandomization` | `void ApplyDomainRandomization(float spatial, float scheduling)` | Applies domain randomization to the observation data |

## Notes
The `ObservationBuilder` class relies heavily on simulation-specific data and functionality from other scripts and classes. The observation streams built by this class are designed to provide a comprehensive view of the simulation state, including machine and AGV status, job progress, and global simulation metrics. The class uses various constants and configuration values to determine the structure and content of the observation data.