# `Simulation/Types/DecisionRequest.cs`
**Language:** C# | **Lines:** 67 | **Last Scanned:** 2026-06-04

## Purpose
This file contains the `DecisionRequest` class, which represents a snapshot of the simulation state when a scheduling decision is required. It provides both shared state and decision-specific fields to support the agent's decision-making process. The class plays a crucial role in the simulation by facilitating the interaction between the agent and the simulation environment.

## Dependencies
* `Assets.Scripts.Simulation.Machines`: provides the `MachineType` enum used in the `DecisionRequest` class.

## Classes
The `DecisionRequest` class represents a scheduling decision request, with no inheritance. 
| Method | Signature | Description |
|--------|-----------|-------------|
| None   |           |             |

## Enums
The `DecisionType` enum defines two categories of scheduling decisions: `Dispatch` and `Routing`.

## Notes
The `DecisionRequest` class has fields that are specific to either `Dispatch` or `Routing` decisions, and only the relevant fields are populated based on the `Type` field. The class uses arrays to store candidate job IDs, durations, and machine IDs, which are parallel to each other. Understanding the context and purpose of each field is essential to using this class effectively.