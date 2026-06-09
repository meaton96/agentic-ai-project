# `Simulation/FailureCoordinator.cs`
**Language:** C# | **Lines:** 1200+ | **Last Scanned:** 2026-06-04

## Purpose
The `FailureCoordinator` class coordinates failure events across the simulation, handling machine failures, repairs, and the resulting job re-routing decisions. It plays a crucial role in maintaining the simulation's integrity and ensuring that jobs are properly re-routed when machines fail. This class is essential for simulating real-world factory operations and testing the robustness of the simulation.

## Dependencies
* `Assets.Scripts.Simulation.Jobs`: provides job data and management functionality
* `Assets.Scripts.Simulation.Machines`: provides machine data and health state management
* `Assets.Scripts.Simulation.AGV`: provides AGV (Automated Guided Vehicle) data and dispatch functionality
* `Assets.Scripts.Simulation.FactoryLayout`: provides factory layout data and machine access functionality
* `Assets.Scripts.Simulation.Stochastic`: provides stochastic event management functionality

## Classes
### FailureCoordinator
Role: coordinates failure events and job re-routing decisions. 
Inheritance: none.
| Method | Signature | Description |
|--------|-----------|-------------|
| Initialize | `void Initialize(JobStore jobs, AGVPool agvPool, FactoryLayoutManager layout, EpisodeTracker tracker, Dictionary<int, double> machineProcessingStartTime, Action<int> onMachineFailedInvalidateDecision, Action<int> refreshLabels)` | Initializes the FailureCoordinator with required dependencies and callbacks. |
| SetSimTime | `void SetSimTime(double simTime)` | Sets the current simulation time reference used for event tracking. |
| HarvestFailureFlags | `void HarvestFailureFlags()` | Checks all machines for failure or repair-complete flags and handles them accordingly. |
| HandleMachineFailure | `void HandleMachineFailure(PhysicalMachine machine)` | Handles all consequences of a machine failure, including re-routing active jobs, canceling pickups and transits, and redirecting or aborting in-flight AGVs. |
| FindBestAlternateMachine | `PhysicalMachine FindBestAlternateMachine(JobData job, int excludeMachineId)` | Finds the best operational alternate machine for a job, excluding a specified machine. |
| RetryDeferredJobs | `void RetryDeferredJobs()` | Re-evaluates deferred jobs and re-queues those whose eligible machines are now operational after a repair completion. |

## Notes
The `FailureCoordinator` class relies heavily on the stochastic event management system to handle machine failures and repairs. The `HarvestFailureFlags` method is only executed when stochastic event management is enabled. The `HandleMachineFailure` method performs a series of steps to re-route jobs and redirect AGVs when a machine fails, ensuring that the simulation remains consistent and realistic.