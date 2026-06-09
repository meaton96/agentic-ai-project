# `Scheduling/Core/Job.cs`
**Language:** C# | **Lines:** 94 | **Last Scanned:** 2026-06-04

## Purpose
This file contains the core classes for job scheduling, specifically the `Operation` and `Job` classes. It provides a foundation for modeling and simulating job execution in a manufacturing or production environment. The classes are designed to be used in conjunction with a discrete event simulator.

## Dependencies
* None (only standard library dependencies)

## Classes
### Operation
Role: Represents a single operation within a job. 
Inheritance: None.
| Method | Signature | Description |
|--------|-----------|-------------|
| Operation | `Operation(int jobId, int opIndex, int machineId, int duration)` | Constructs a fully specified, immutable operation. |

### Job
Role: Represents an ordered sequence of operations. 
Inheritance: None.
| Method | Signature | Description |
|--------|-----------|-------------|
| Job | `Job(int id, Operation[] operations, double arrivalTime = 0)` | Constructs a job with a fixed operation sequence and an optional arrival time. |

## Notes
The `Operation` class is immutable after construction, except for `StartTime` and `EndTime`, which are set by the simulator during the simulation run. The `Job` class tracks execution progress via the `NextOperationIndex` property, which advances each time an operation completes. The `IsComplete` property is derived directly from this index and requires no additional state.