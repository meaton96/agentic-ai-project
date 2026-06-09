# `Simulation/Machines/MachineHealthState.cs`
**Language:** C# | **Lines:** 26 | **Last Scanned:** 2026-06-04

## Purpose
This file defines an enumeration representing the health lifecycle state of a physical machine in the simulation. It is used by `PhysicalMachine` and `SimulationBridge` to manage machine availability and routing decisions. The state transitions are driven externally by `SimulationBridge` methods.

## Enums
The `MachineHealthState` enum represents the health state of a machine, with the following values:
| Value | Description |
|-------|-------------|
| Operational | Normal processing state, TTF countdown is running. |
| Failed | TTF has expired, failed flag is set, and repair duration is sampled. |
| Repairing | Repair is in progress, remaining repair time is counting down. |

## Notes
The state transitions are driven externally by `SimulationBridge` methods, including `PhysicalMachine.AcknowledgeFailure` and `PhysicalMachine.AcknowledgeRepairComplete`. The `MachineHealthState` is also encoded as a 4th channel in the 64×64 spatial occupancy tensor for the RL observation.