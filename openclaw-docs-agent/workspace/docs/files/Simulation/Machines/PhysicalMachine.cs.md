# `Simulation/Machines/PhysicalMachine.cs`
**Language:** C# | **Lines:** 12000+ | **Last Scanned:** 2026-06-04

## Purpose
This file represents a physical processing unit within the factory simulation, managing processing timers, state flags, and stochastic failure and repair mechanisms. It plays a crucial role in the simulation by providing a realistic model of machine behavior. The class is designed to be polled by the SimulationBridge, which drives the factory lifecycle.

## Dependencies
* `UnityEngine`: for Unity-specific functionality and components
* `Assets.Scripts.Simulation.Jobs`: for job-related data and functionality
* `Assets.Scripts.Simulation.Stochastic`: for stochastic event management and sampling
* `Assets.Scripts.Simulation.Logging`: for logging and debugging purposes
* `Assets.Scripts.Simulation.Channels`: for communication and data exchange between components

## Classes
### PhysicalMachine
Represents a physical machine in the simulation, inheriting from `MonoBehaviour`.
| Method | Signature | Description |
|--------|-----------|-------------|
| `CanProcess` | `bool CanProcess(MachineType opType)` | Checks if the machine can process a specific operation type |
| `Initialize` | `void Initialize(int id, MachineType primary, IEnumerable<MachineType> capabilities = null)` | Initializes the machine with its identity, capabilities, and visual layer |
| `InitializeStochastic` | `void InitializeStochastic()` | Seeds the machine's time-to-failure countdown for the current episode |
| `StartJob` | `void StartJob(int jobId, float duration, JobVisual visual = null)` | Begins processing a specific job for a defined duration |
| `DEBUG_ForceFailure` | `void DEBUG_ForceFailure()` | Forces an immediate machine failure for editor testing (only available in Unity Editor) |

## Notes
The `PhysicalMachine` class uses a stochastic event manager to simulate machine failures and repairs, adding realism to the simulation. The `InitializeStochastic` method is used to seed the machine's time-to-failure countdown, and the `StartJob` method begins processing a job while updating the machine's state and visual representation. The class also includes a debug method to force an immediate machine failure for testing purposes.