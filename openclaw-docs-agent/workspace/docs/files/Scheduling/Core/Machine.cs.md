# `Scheduling/Core/Machine.cs`
**Language:** C# | **Lines:** 88 | **Last Scanned:** 2026-06-04

## Purpose
This file contains the `Machine` class, which represents a single machine on the shop floor in a simulation. It maintains the machine's current state, the operation it is actively processing, and a queue of operations that have arrived but cannot yet start. The class is used by the `DESSimulator` to manage machine state transitions and operation processing.

## Dependencies
* `System.Collections.Generic`: provides the `List<T>` class used for the `WaitingQueue` property.

## Classes
### Machine
Represents a single machine on the shop floor. Does not inherit from any base class.
| Method | Signature | Description |
|--------|-----------|-------------|
| StartProcessing | `void StartProcessing(Operation op, double currentTime)` | Transitions the machine to the busy state and begins processing an operation. |
| FinishProcessing | `void FinishProcessing()` | Transitions the machine to the idle state after an operation completes. |

## Notes
The `Machine` class is designed to be used by the `DESSimulator` to manage machine state transitions and operation processing. The `StartProcessing` and `FinishProcessing` methods are used to update the machine's state and operation in progress, while the `WaitingQueue` property is used to store operations that have arrived but cannot yet start. The `DESSimulator` is responsible for selecting the next operation from the `WaitingQueue` and starting it on the machine.