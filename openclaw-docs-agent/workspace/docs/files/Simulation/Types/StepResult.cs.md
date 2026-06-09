# `Simulation/Types/StepResult.cs`
**Language:** C# | **Lines:** 32 | **Last Scanned:** 2026-06-04

## Purpose
This file defines the `StepResult` struct, which represents the result of a simulation step in a factory job scheduling problem. It encapsulates the reward signal, episode termination flag, and the next decision context for the agent. The `StepResult` struct plays a crucial role in the simulation loop, providing feedback to the agent after each discrete event.

## Classes
* `StepResult`: Represents the result of a simulation step, with no inheritance.
| Field | Type | Description |
|--------|-----------|-------------|
| Reward | float | Reward signal for the agent |
| Done | bool | Episode termination flag |
| NextDecision | DecisionRequest | Next decision context for the agent |
| CurrentMakespan | double | Current makespan at the time of this step |
| OperationsCompleted | int | Total number of operations completed |

## Notes
The `StepResult` struct is populated by the simulation after each discrete event, such as machine completion or AGV delivery. It provides a concise summary of the current state of the simulation, allowing the agent to make informed decisions about its next action. The `DecisionRequest` type is assumed to be defined elsewhere in the project.