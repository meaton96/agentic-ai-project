# `Scheduling/Core/EventQueue.cs`
**Language:** C# | **Lines:** 137 | **Last Scanned:** 2026-06-04

## Purpose
This file implements a simulation event queue, providing a priority queue for scheduling events in a discrete event simulation. The `EventQueue` class manages a sorted set of `SimEvent` instances, ensuring that events are processed in the correct order. This component is crucial for the project's simulation framework, enabling efficient and accurate event handling.

## Dependencies
* `System.Collections.Generic`: provides the `SortedSet` class, which is used to implement the event queue.
* `System`: provides basic data types and interfaces, such as `IComparable`, which is implemented by the `SimEvent` class.

## Classes
### SimEvent
Role: represents a single scheduled event in the simulation timeline.
Inheritance: implements `IComparable<SimEvent>`.
| Method | Signature | Description |
|--------|-----------|-------------|
| CompareTo | `int CompareTo(SimEvent other)` | Compares this event to another for priority queue ordering. |
### EventQueue
Role: manages a priority queue for simulation events, ordered by time with FIFO tiebreaking.
Inheritance: none.
| Method | Signature | Description |
|--------|-----------|-------------|
| Enqueue | `SimEvent Enqueue(double time, EventType type, int jobId = -1, int machineId = -1)` | Creates and enqueues a new `SimEvent` at the specified simulation time. |
| Dequeue | `SimEvent Dequeue()` | Removes and returns the earliest-scheduled event in the queue. |
| Peek | `SimEvent Peek()` | Returns the earliest-scheduled event without removing it from the queue. |
| Clear | `void Clear()` | Removes all events from the queue and resets the sequence counter to zero. |

## Notes
The `EventQueue` class uses a `SortedSet` to maintain the event queue, which ensures efficient insertion and removal of events. The `SimEvent` class implements `IComparable` to enable sorting based on the event time and sequence number. The sequence number is used as a FIFO tiebreaker to ensure that events with the same time are processed in the order they were enqueued.