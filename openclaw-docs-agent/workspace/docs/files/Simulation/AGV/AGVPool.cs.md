# `Simulation/AGV/AGVPool.cs`
**Language:** C# | **Lines:** 123 | **Last Scanned:** 2026-06-04

## Purpose
This file contains the `AGVPool` class, which manages the lifecycle and retrieval of the AGV fleet in a simulation. It serves as a centralized container and factory, allowing the `SimulationBridge` orchestrator to query and dispatch available units. The class is responsible for initializing, clearing, and retrieving AGV units.

## Dependencies
* `UnityEngine`: for Unity-specific functionality and types
* `Assets.Scripts.Simulation.FactoryLayout`: for factory layout management
* `Assets.Scripts.Simulation.Logging`: for logging and debugging purposes

## Classes
The `AGVPool` class is a `MonoBehaviour` that manages the AGV fleet.
| Method | Signature | Description |
|--------|-----------|-------------|
| `Awake` | `void Awake()` | Initializes the `AGVPool` instance |
| `InitializeFleet` | `void InitializeFleet(int fleetSize = 3)` | Destroys the existing fleet and instantiates new AGV units |
| `ClearFleet` | `void ClearFleet()` | Destroys all AGVs in the fleet and clears the pool |
| `GetParkingPosition` | `Vector3 GetParkingPosition(int agvId)` | Retrieves the designated world-space parking coordinate for a specific AGV |
| `GetIdleAGV` | `AGVController GetIdleAGV()` | Locates the first unit that is currently in a strictly `Idle` state |
| `GetAvailableAGV` | `AGVController GetAvailableAGV()` | Identifies the best candidate for a new task dispatch |
| `GetPreDispatchedAGV` | `AGVController GetPreDispatchedAGV(int jobId)` | Returns the AGV unit assigned to a specific job ID via pre-dispatch |

## Notes
The `AGVPool` class uses a singleton pattern to ensure only one instance exists. The `InitializeFleet` method calculates parking positions based on the `FactoryLayoutManager` coordinates and spawns units in a linear arrangement. The `GetAvailableAGV` method performs a two-pass search to optimize travel time.