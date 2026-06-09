# `Scheduling/Core/DispatchingRules.cs`
**Language:** C# | **Lines:** 104 | **Last Scanned:** 2026-06-04

## Purpose
This file implements dispatching rules for machine queue selection in a scheduling system. It provides a set of rules to determine the next operation to process from a machine's waiting queue. The rules are designed to optimize the scheduling process based on various heuristics.

## Dependencies
* `System.Collections.Generic`: for using generic collections such as `List<T>`
* `System.Linq`: for using LINQ queries to sort and filter operations

## Classes
* `DispatchingRules`: a static class that provides methods for selecting the next operation to process based on a given dispatching rule.
  + Inheritance: none
  | Method | Signature | Description |
  |--------|-----------|-------------|
  | `SelectNext` | `Operation SelectNext(List<Operation> waitingOps, Job[] allJobs, DispatchingRule rule)` | Selects the next operation to process from a machine's waiting queue based on the given dispatching rule |
  | `RemainingWork` | `int RemainingWork(Operation op, Job[] allJobs)` | Computes the total remaining processing time for a job from a given operation onward |

## Enums
* `DispatchingRule`: an enumeration of dispatching rules, including SPT_SMPT, SPT_SRWT, LPT_MMUR, LPT_SMPT, SRT_SRWT, SRT_SMPT, LRT_MMUR, SDT_SRWT, and MOR.

## Notes
The dispatching rules are designed to optimize the scheduling process based on various heuristics, such as shortest processing time, longest processing time, and most operations remaining. The rules are implemented using LINQ queries to sort and filter operations. The `RemainingWork` method is used to compute the total remaining processing time for a job from a given operation onward, which is used as a sort key for some dispatching rules.