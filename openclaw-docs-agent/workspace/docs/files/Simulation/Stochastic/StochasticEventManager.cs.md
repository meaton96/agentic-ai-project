# `Simulation/Stochastic/StochasticEventManager.cs`
**Language:** C# | **Lines:** 12000+ | **Last Scanned:** 2026-06-04

## Purpose
This file implements the `StochasticEventManager` class, a singleton responsible for managing stochastic events in a simulation. It provides a seeded random number generator and sampling methods for various distributions, ensuring deterministic and reproducible simulation runs. The class plays a crucial role in introducing randomness and uncertainty into the simulation.

## Dependencies
* `UnityEngine`: for MonoBehaviour inheritance and Unity integration
* `Assets.Scripts.Simulation.Types`: for custom simulation-related types, such as `StochasticConfig` and `FJSSPConfig`
* `Assets.Scripts.Simulation.Logging`: for logging functionality

## Classes
### StochasticEventManager
Manages stochastic events in the simulation. Inherits from `MonoBehaviour`.
| Method | Signature | Description |
|--------|-----------|-------------|
| Awake | `void Awake()` | Initializes the singleton instance |
| Initialize | `void Initialize(FJSSPConfig config)` | Re-initializes the stochastic event manager with a new simulation config |
| SampleMachineTTF | `float SampleMachineTTF()` | Samples the time-to-failure for a machine from a Weibull distribution |
| SampleMachineRepair | `float SampleMachineRepair()` | Samples the repair duration for a machine from a LogNormal distribution |
| SampleAGVTTF | `float SampleAGVTTF()` | Samples the time-to-failure for an AGV from a Weibull distribution |
| SampleAGVRepair | `float SampleAGVRepair()` | Samples the repair duration for an AGV from a LogNormal distribution |
| SampleInterArrivalTime | `float SampleInterArrivalTime()` | Samples the time until the next job arrival from an Exponential distribution |
| SampleWeibull | `float SampleWeibull(float k, float lambda)` | Generates a sample from a Weibull distribution |
| SampleLogNormal | `float SampleLogNormal(float mu, float sigma)` | Generates a sample from a LogNormal distribution |
| SampleExponential | `float SampleExponential(float lambda)` | Generates a sample from an Exponential distribution |
| SampleStandardNormal | `double SampleStandardNormal()` | Generates a standard normal random variate |
| NextNonZeroUniform | `double NextNonZeroUniform()` | Generates a non-zero uniform random variate |

## Notes
The `StochasticEventManager` class uses a seeded random number generator to ensure deterministic simulation runs. The `Initialize` method re-seeds the RNG and caches the stochastic parameters from the provided config. The sampling methods use various distributions (Weibull, LogNormal, Exponential) to introduce randomness and uncertainty into the simulation. The class also provides convenience properties (e.g., `MachineFailuresEnabled`) to quickly check the status of stochastic features.