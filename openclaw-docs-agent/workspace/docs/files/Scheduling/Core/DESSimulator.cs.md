# `Scheduling/Core/DESSimulator.cs`
**Language:** C# | **Lines:** 12000+ | **Last Scanned:** 2026-06-04

## Purpose
This file contains the `DESSimulator` class, a pure discrete-event simulator for the Job Shop Problem (JSP) validation. It provides a standalone simulation environment without Unity dependencies, allowing for benchmark validation and later integration with a reinforcement learning (RL) layer. The simulator manages job scheduling, machine operations, and event processing.

## Dependencies
* `Assets.Scripts.Scheduling.Data`: provides data structures and types for job shop scheduling, such as `Job`, `Machine`, and `Operation`.
* `System.Collections.Generic`: used for generic collections, like lists and queues, to manage simulator state.

## Classes
### DESSimulator
Role: discrete-event simulator for job shop scheduling.
Inheritance: none.
| Method | Signature | Description |
|--------|-----------|-------------|
| ProcessNextEvent | `SimStepResult ProcessNextEvent()` | Processes a single event from the queue, returning a `SimStepResult` indicating whether the simulation needs an external dispatch decision, can continue autonomously, or has finished. |
| HandleOperationCompleteStepped | `void HandleOperationCompleteStepped(SimEvent evt)` | Stepped variant of `HandleOperationComplete`, which pauses for an external decision instead of auto-dispatching. |
| TryStartNextOnMachineStepped | `void TryStartNextOnMachineStepped(Machine machine)` | Stepped variant of `TryStartNextOnMachine`, which sets `WaitingForDecision` and `PendingDecisionMachineId` instead of auto-selecting the next operation. |
| ApplyDecision | `void ApplyDecision(DispatchingRule rule)` | Applies an external dispatch decision, selecting the next operation from the pending machine's waiting queue, starting processing, and enqueuing the corresponding completion event. |
| ResetSteppedState | `void ResetSteppedState()` | Resets the stepped-execution state, clearing `WaitingForDecision` and `PendingDecisionMachineId`. |
| LoadInstance | `void LoadInstance(TaillardInstance instance)` | Loads a Taillard instance, initializing jobs and machines, and preparing the simulator for execution. |
| Reset | `void Reset()` | Resets all simulator state without reloading the instance, clearing the event queue, resetting time and statistics counters, and re-enqueuing job arrival events. |

## Notes
The `DESSimulator` class uses a `WaitingForDecision` flag to pause the simulation when a machine requires a dispatch decision, allowing for external intervention and decision-making. The `ApplyDecision` method is used to apply an external dispatch decision, resuming the simulation. The simulator also provides a `ResetSteppedState` method to reset the stepped-execution state, and a `LoadInstance` method to load a Taillard instance and initialize the simulator.