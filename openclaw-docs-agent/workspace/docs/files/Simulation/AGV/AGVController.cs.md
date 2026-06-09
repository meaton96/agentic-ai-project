# `Simulation/AGV/AGVController.cs`
**Language:** C# | **Lines:** 12000+ | **Last Scanned:** 2026-06-04

## Purpose
This file contains the `AGVController` class, which controls the navigation and physical movement of a single Automated Guided Vehicle (AGV) in a simulation environment. The class acts as a state-driven controller, managing physical movement and setting status flags for a central supervisor to process. It plays a crucial role in the project by enabling the simulation of AGV operations in a factory setting.

## Dependencies
* `UnityEngine`: provides core Unity functionality for game object manipulation and navigation.
* `UnityEngine.AI`: enables the use of NavMeshAgent for pathfinding and navigation.
* `Assets.Scripts.Simulation.Machines`: provides access to machine-related functionality and data.
* `Assets.Scripts.Simulation.FactoryLayout`: allows interaction with the factory layout and traffic management.
* `Assets.Scripts.Simulation.Logging`: facilitates logging and debugging.

## Classes
### AGVController
Role: Controls AGV navigation and movement. 
Inherits from: `MonoBehaviour`.
| Method | Signature | Description |
|--------|-----------|-------------|
| `Initialize` | `public void Initialize(int id)` | Sets up the AGV identity and initializes navigation components. |
| `ClearFlags` | `public void ClearFlags()` | Resets all completion flags and delivery metadata. |
| `ResetEpisodeStats` | `public void ResetEpisodeStats()` | Zeros all per-episode statistics. |
| `GetRecord` | `public Assets.Scripts.Simulation.Types.AGVRecord GetRecord(double makespan)` | Builds an AGVRecord snapshot from accumulated stats at episode end. |
| `SetIdleCallback` | `public void SetIdleCallback(System.Action callback)` | Sets the callback to be invoked when the AGV becomes idle. |
| `AbortTransit` | `public void AbortTransit()` | Aborts an in-progress dropoff when no redirect target is available. |
| `CancelPickup` | `public void CancelPickup()` | Cancels an in-progress pickup route when the destination machine fails. |
| `RedirectDropoff` | `public void RedirectDropoff(Vector3 newDropoffPos, PhysicalMachine newTarget, JobVisual visual)` | Redirects an AGV that is already carrying a job to a new dropoff machine. |
| `PreDispatch` | `public void PreDispatch(int jobId, Vector3 pickupPos, PhysicalMachine source)` | Commands the AGV to move to a pickup zone before a job is finished. |

## Notes
The `AGVController` class follows an "orchestrator-flag" pattern, where it manages physical movement and sets status flags for a central supervisor to process, rather than triggering job state transitions directly. This design allows for a decoupling of AGV movement and job state management, enabling more flexible and scalable simulation scenarios.