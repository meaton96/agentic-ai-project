# `FailureTestDriver.cs`
**Language:** C# | **Lines:** 64 | **Last Scanned:** 2026-06-04

## Purpose
The `FailureTestDriver.cs` file contains a class that drives failure testing for physical machines in a simulation. It allows for forcing a failure on a specific machine and monitors its health state. This class plays a crucial role in testing and debugging the simulation's failure handling mechanisms.

## Dependencies
* `UnityEngine`: provides core Unity functionality
* `Assets.Scripts.Simulation.Machines`: provides the `PhysicalMachine` class, which is used for failure testing
* `System.Linq`: provides LINQ functionality for querying collections of machines

## Classes
The `FailureTestDriver` class is a `MonoBehaviour` that drives failure testing for physical machines.
| Method | Signature | Description |
|--------|-----------|-------------|
| `Update` | `void Update()` | Updates the driver's state and checks for failure triggers |
| `ForceFailure` | `void ForceFailure()` | Forces a failure on the target machine |
| `FindFirstMachineById` | `PhysicalMachine FindFirstMachineById(int id)` | Finds the first machine with the specified ID |

## Notes
The `FailureTestDriver` class uses reflection to access the private `_ttfCountdown` field of the `PhysicalMachine` class. This is done to monitor the time-to-failure (TTF) countdown. The class also uses the new input system to detect when the 'P' key is pressed, which triggers a forced failure on the target machine. The `FindFirstMachineById` method uses `FindObjectsByType` with `FindObjectsSortMode.None` to efficiently find machines by ID.