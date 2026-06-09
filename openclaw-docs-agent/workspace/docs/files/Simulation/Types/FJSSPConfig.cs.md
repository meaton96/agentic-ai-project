# `Simulation/Types/FJSSPConfig.cs`
**Language:** C# | **Lines:** 138 | **Last Scanned:** 2026-06-04

## Purpose
This file defines the configuration parameters for a Flexible Job Shop Scheduling Problem (FJSSP) instance. It governs job shop generation, including machine topology, processing time ranges, operation counts, and stochastic disruption settings. The class is used by `ConfigLoader` to parse from JSON and by `FJSSPJobGenerator` to create problem instances.

## Dependencies
* `Assets.Scripts.Simulation.Machines`: Provides machine-related types and functionality.
* `Assets.Scripts.Simulation.Stochastic`: Enables stochastic disruption settings for the simulation.

## Classes
The `FJSSPConfig` class represents the configuration for an FJSSP instance. It does not inherit from any base class.
| Method | Signature | Description |
|--------|-----------|-------------|
| CloneWithSeed | `FJSSPConfig CloneWithSeed(int newSeed)` | Returns a deep clone with a new seed, used to generate varied instances across repeated runs. |

## Notes
The `Stochastic` property is shared by reference in the `CloneWithSeed` method, meaning changes to the stochastic configuration in the cloned instance will affect the original instance. The `MachineFlexibilityProbability` property controls the probability of machines gaining secondary capabilities, allowing for flexible machine configurations.