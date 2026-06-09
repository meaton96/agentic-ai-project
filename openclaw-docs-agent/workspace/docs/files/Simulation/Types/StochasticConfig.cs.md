# `Simulation/Types/StochasticConfig.cs`
**Language:** C# | **Lines:** 138 | **Last Scanned:** 2026-06-04

## Purpose
This file defines the `StochasticConfig` class, which holds parameters for stochastic disruptions in a simulation. It plays a crucial role in introducing randomness and unpredictability into the simulation, making it more realistic. The class is used in conjunction with the `FJSSPConfig` and `StochasticEventManager` to control the simulation's behavior.

## Dependencies
* None notable beyond standard library dependencies.

## Classes
The `StochasticConfig` class represents a set of stochastic disruption parameters.
* Inheritance: None
| Method | Signature | Description |
|--------|-----------|-------------|
| (Properties) | `Tag` | Gets a descriptive tag for log output and CSV labelling, composed from active disruption types. |
| (Properties) | `AnyEnabled` | Gets a boolean indicating whether any disruption source is active. |

## Notes
The `StochasticConfig` class is designed to be attached to a `FJSSPConfig` via JSON, allowing for easy configuration of stochastic disruptions in the simulation. The class includes various parameters to control machine failures, repair times, AGV failures, and dynamic job arrivals, providing a high degree of customization for the simulation.