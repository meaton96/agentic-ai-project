# `Scheduling/Validation/mirror_des.py`
**Language:** Python | **Lines:** 12000+ | **Last Scanned:** 2026-06-04

## Purpose
This Python script implements a Discrete Event Simulator (DES) mirroring the C# DESSimulator exactly, ensuring any makespan delta between the two is a definitive indicator of a real bug. It runs simulation instances with various dispatching rules and exports makespans and schedules for comparison. The script is used for validation and testing purposes.

## Dependencies
* `dataclasses`: for creating immutable data structures like `SimEvent`, `Operation`, `Job`, and `Machine`.
* `enum`: for defining enumeration types like `EventType` and `MachineState`.
* `heapq`: for implementing priority queues.
* `argparse`: for parsing command-line arguments.

## Classes
### EventType
One-line role description: Discriminated union of simulation event types.
Inheritance note: Inherits from `Enum`.
No methods to document.

### SimEvent
One-line role description: Immutable simulation event with (time, sequence) ordering.
Inheritance note: No inheritance.
No methods to document.

### Operation
One-line role description: Single operation within a job.
Inheritance note: No inheritance.
No methods to document.

### Job
One-line role description: Ordered sequence of operations representing a single job.
Inheritance note: No inheritance.
| Method | Signature | Description |
|--------|-----------|-------------|
| `is_complete` | `-> bool` | Checks if all operations have been processed. |
| `current_operation` | `-> Optional[Operation]` | Returns the next operation awaiting dispatch, or `None` if the job is complete. |

### Machine
One-line role description: Single shop-floor machine with a waiting queue.
Inheritance note: No inheritance.
| Method | Signature | Description |
|--------|-----------|-------------|
| `start_processing` | `(op: Operation, current_time: float) -> None` | Transitions to `BUSY` and stamps `start_time` / `end_time` on `op`. |
| `finish_processing` | `-> None` | Transitions to `IDLE` and clears `current_op`. |

### MachineState
One-line role description: Operational states for a machine.
Inheritance note: Inherits from `Enum`.
No methods to document.

## Functions
| Function | Signature | Description |
|----------|-----------|-------------|
| `remaining_work` | `(op: Operation, jobs: list[Job]) -> int` | Computes total remaining processing time for a job from `op` onward. |
| `remaining_ops` | `(op: Operation, jobs: list[Job]) -> int` | Computes the number of unprocessed operations remaining for a job from `op` onward. |
| `rule_spt` | `(waiting: list[Operation], jobs: list[Job]) -> Operation` | Selects the operation with the smallest duration. |
| `rule_lpt` | `(waiting: list[Operation], jobs: list[Job]) -> Operation` | Selects the operation with the largest duration. |
| `rule_mwr` | `(waiting: list[Operation], jobs: list[Job]) -> Operation` | Selects the operation whose job has the most remaining work. |
| `rule_mor` | `(waiting: list[Operation], jobs: list[Job]) -> Operation` | Selects the operation whose job has the most operations left. |
| `rule_fcfs` | `(waiting: list[Operation], jobs: list[Job]) -> Operation` | Returns the first operation in the queue. |

## Notes
The script uses a priority queue to manage simulation events, ensuring that events are processed in the correct order. The `DESSimulator` class is responsible for handling events, dispatching operations, and managing machine queues. The script exports makespans and schedules to CSV and JSON files for comparison with C# results.