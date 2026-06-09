# `Simulation/Jobs/FJSSPJobDefinition.cs`
**Language:** C# | **Lines:** 43 | **Last Scanned:** 2026-06-04

## Purpose
This file defines a static job specification for the Flexible Job Shop Scheduling Problem (FJSSP). It serves as a blueprint for job initialization, converted into live `JobData` instances at runtime by `JobStore`. The class provides essential details about a job, including its unique identifier, arrival time, operation sequence, and eligible machines.

## Dependencies
* `Assets.Scripts.Simulation.Machines`: provides the `MachineType` enum used in the `OperationSequence` field.

## Classes
* `FJSSPJobDefinition`: defines a job specification for FJSSP, with no inheritance.
| Method | Signature | Description |
|--------|-----------|-------------|
| None   |           | This class has no methods, only public fields. |

## Notes
The `FJSSPJobDefinition` class is a data container, holding job-specific information. Its fields are used to initialize `JobData` instances at runtime. The `EligibleMachinesPerOp` field is an array of dictionaries, where each dictionary maps a machine ID to its processing time for a specific operation. This allows for flexible job scheduling and machine allocation.