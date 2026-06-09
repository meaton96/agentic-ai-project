# `Simulation/Machines/MachineState.cs`
**Language:** C# | **Lines:** 29 | **Last Scanned:** 2026-06-04

## Purpose
This file defines an enumeration representing the operational state of a machine within the simulation. It is used by `MachineVisual` to determine indicator colors and status text, and by `PhysicalMachine` to track processing lifecycle. The enum values map to corresponding materials in the `indicatorMaterials` array on `MachineVisual`.

## Enums
The `MachineState` enum has the following values:
| Value | Description |
|-------|-------------|
| Idle  | Machine is idle and available to accept new jobs. |
| Busy  | Machine is actively processing a job. |
| Blocked | Machine has finished processing but cannot release the job. |
| Failed | Machine has experienced a failure and is non-operational. |
| Repair | Machine is undergoing repair and is unavailable for work. |

## Notes
The `MachineState` enum is used to track the state of machines in the simulation, and its values are used to determine the visual representation and behavior of machines. The enum values are designed to be used in a state machine-like fashion, with each value representing a distinct state in the machine's lifecycle.