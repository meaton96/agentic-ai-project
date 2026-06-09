# `Simulation/DecisionCoordinator.cs`
**Language:** C# | **Lines:** 177 | **Last Scanned:** 2024-02-20

## Purpose
The `DecisionCoordinator` class is responsible for making routing and dispatching decisions in a factory simulation. It manages the lifecycle of decision requests, including job routing to eligible machines and dispatching jobs to available machines. This class plays a central role in the simulation, ensuring efficient allocation of resources.

## Dependencies
* `Assets.Scripts.Simulation.Jobs`: provides job data and state information
* `Assets.Scripts.Simulation.FactoryLayout`: provides machine information and layout management
* `Assets.Scripts.Simulation.Types`: defines simulation-related data types
* `Assets.Scripts.Simulation.Logging`: handles logging and debugging output

## Classes
The `DecisionCoordinator` class is the primary class in this file.
* Role: Central coordinator for making routing and dispatching decisions in the factory simulation.
* Inheritance: None
| Method | Signature | Description |
|--------|-----------|-------------|
| Initialize | `void Initialize(JobStore jobs, FactoryLayoutManager layout, Func<double> getSimTime, Func<int> getDecisionCount, Action incrementDecisionCount)` | Initializes the coordinator with required dependencies. |
| FindNextDecision | `DecisionRequest FindNextDecision()` | Determines the next decision to be made by the simulation, prioritizing routing decisions for jobs needing machine assignment. |
| BuildRoutingDecision | `DecisionRequest BuildRoutingDecision(JobData job)` | Builds a routing decision request for a job that needs to be assigned to an eligible machine. |
| BuildDispatchDecision | `DecisionRequest BuildDispatchDecision(int machineId)` | Builds a dispatch decision request for a machine that has jobs waiting in its queue. |

## Notes
The `DecisionCoordinator` class uses dependency injection to receive required dependencies, such as the job store, factory layout manager, and simulation time retrieval functions. This design allows for loose coupling and easier testing of the class. The class also uses delegates to retrieve the current simulation time and decision count, providing flexibility in how these values are obtained.