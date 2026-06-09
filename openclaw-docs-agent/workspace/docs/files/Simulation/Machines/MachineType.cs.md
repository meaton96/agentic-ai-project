# `Simulation/Machines/MachineType.cs`
**Language:** C# | **Lines:** 27 | **Last Scanned:** 2026-06-04

## Purpose
This file defines an enumeration of machine types used in a factory simulation. The `MachineType` enum represents different manufacturing operations and is used for job routing, capability matching, and scheduling decisions. It provides a way to identify and distinguish between various machines in the simulation.

## Enums
The `MachineType` enum has the following values:
| Value | Description |
|-------|-------------|
| Mill  | Milling machine — removes material from a workpiece using rotating cutters. |
| Lathe | Lathe — rotates the workpiece to cut with a stationary tool. |
| Weld  | Welding station — joins workpieces through welding processes. |
| Inspect | Inspection station — performs quality checks on workpieces. |
| Assemble | Assembly station — assembles multiple components into a final product. |

## Notes
The `MachineType` enum is used to associate a unique color with each machine type for visual identification in the `MachineVisual` component. This enum plays a crucial role in the simulation's logic, enabling the system to make informed decisions about job routing, capability matching, and scheduling.