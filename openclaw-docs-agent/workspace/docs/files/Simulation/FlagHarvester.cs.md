# `Simulation/FlagHarvester.cs`
**Language:** C# | **Lines:** 278 | **Last Scanned:** Not specified

## Purpose
The `FlagHarvester` class coordinates the flow of jobs through the factory simulation by harvesting state-change flags from machines, AGVs, and the job scheduler. It orchestrates job state transitions, AGV dispatching, pre-dispatch logic, and timing statistics collection. This class plays a crucial role in managing the simulation's workflow and ensuring efficient job processing.

## Dependencies
* `Assets.Scripts.Simulation.Jobs`: provides job data and state management
* `Assets.Scripts.Simulation.Machines`: provides machine data and state management
* `Assets.Scripts.Simulation.AGV`: provides AGV data and state management
* `Assets.Scripts.Simulation.FactoryLayout`: provides factory layout data and queries
* `Assets.Scripts.Simulation.Logging`: provides logging functionality for simulation events

## Classes
### FlagHarvester
The `FlagHarvester` class is responsible for harvesting state-change flags and managing job state transitions. It does not inherit from any other class.
| Method | Signature | Description |
|--------|-----------|-------------|
| Initialize | `void Initialize(JobStore jobs, AGVPool agvPool, FactoryLayoutManager layout, EpisodeTracker tracker, Dictionary<int, double> processingStartTimes)` | Initializes the FlagHarvester with required dependencies |
| SetSimTime | `void SetSimTime(double simTime)` | Sets the current simulation time reference |
| HarvestMachineFlags | `void HarvestMachineFlags()` | Harvests completion flags from all machines and updates job state |
| HarvestAlmostDoneFlags | `void HarvestAlmostDoneFlags(int preDispatchLeadTime)` | Harvests "almost done" flags from machines and initiates AGV pre-dispatch |
| HarvestAGVFlags | `void HarvestAGVFlags()` | Harvests state-change flags from all AGVs and updates job state |
| AssignAGVs | `void AssignAGVs()` | Assigns available AGVs to jobs that are waiting for pickup |
| RefreshMachineLabels | `void RefreshMachineLabels(int machineId)` | Refreshes the queue count labels on a specified machine |

## Notes
The `FlagHarvester` class relies on the `JobStore`, `AGVPool`, `FactoryLayoutManager`, and `EpisodeTracker` classes to manage job data, AGV state, factory layout, and simulation statistics, respectively. The class uses a dictionary to track machine processing start times and updates job state based on machine and AGV flags. The `AssignAGVs` method assigns available AGVs to jobs based on their location and target destination, taking into account machine operational health.