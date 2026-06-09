# `Simulation/DispatchingEngine.cs`
**Language:** C# | **Lines:** 550 | **Last Scanned:** 2026-06-04

## Purpose
The `DispatchingEngine` class provides static methods for evaluating and applying dispatching rules in the Flexible Job Shop Scheduling Problem (FJSSP) simulation. It maps action indices to dispatching rules, selects jobs for machines, selects machines for jobs, and computes job metrics such as remaining work. This class plays a crucial role in the simulation by enabling the application of various dispatching rules to optimize job scheduling.

## Dependencies
* `Assets.Scripts.Simulation.Jobs`: Provides job-related data and functionality.
* `Assets.Scripts.Simulation.Types`: Defines types used in the simulation, including `DispatchingRule`.
* `System.Collections.Generic`: Used for generic collections, such as lists.
* `System.Linq`: Enables LINQ queries for data manipulation.

## Classes
### DispatchingEngine
One-line role description: Provides static methods for dispatching rule evaluation and application.
Inheritance note: None.
| Method | Signature | Description |
|--------|-----------|-------------|
| RuleForIndex | `public static DispatchingRule RuleForIndex(int index)` | Retrieves the dispatching rule associated with the given index. |
| IndexForRule | `public static int IndexForRule(DispatchingRule rule)` | Retrieves the index of the given dispatching rule within the registered rule array. |
| SelectJob | `public static int SelectJob(int actionIndex, int machineId, JobStore jobs, double simTime)` | Selects the best job for a given machine based on the dispatching rule specified by actionIndex. |
| SelectMachine | `public static int SelectMachine(int actionIndex, DecisionRequest req)` | Selects the best machine from candidate machines based on the dispatching rule specified by actionIndex. |
| GetRemainingWork | `public static float GetRemainingWork(int jobId, JobStore jobs)` | Computes the total remaining work for a job. |
| ArgMin | `private static int ArgMin(List<int> ids, Func<int, float> score)` | Finds the element in the list with the minimum score value. |
| ArgMax | `private static int ArgMax(List<int> ids, Func<int, float> score)` | Finds the element in the list with the maximum score value. |
| ArgMinIdx | `private static int ArgMinIdx(float[] v)` | Finds the index of the element with the minimum value in a float array. |
| ArgMaxIdx | `private static int ArgMaxIdx(float[] v)` | Finds the index of the element with the maximum value in a float array. |

## Notes
The `DispatchingEngine` class uses a switch statement to apply different dispatching rules based on the `actionIndex`. The `Random` rule is handled by re-sampling a specific non-random rule at decision time. The class also uses LINQ queries to manipulate data and compute job metrics. The `GetRemainingWork` method computes the total remaining work for a job by summing the minimum processing times for each remaining operation.