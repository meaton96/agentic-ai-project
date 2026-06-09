# `Simulation/Jobs/FJSSPJobGenerator.cs`
**Language:** C# | **Lines:** 189 | **Last Scanned:** (not provided)

## Purpose
This file contains the `FJSSPJobGenerator` class, which generates Flexible Job Shop Scheduling Problem (FJSSP) instances with randomized job structures and machine eligibility. It plays a crucial role in simulating production environments and testing scheduling algorithms. The class provides methods to generate jobs with varying operation sequences, processing times, and arrival times.

## Dependencies
* `Assets.Scripts.Simulation.Machines`: provides `MachineType` enum and related functionality
* `Assets.Scripts.Simulation.Types`: provides `FJSSPConfig` and `FJSSPJobDefinition` classes
* `UnityEngine`: used for random number generation and other utility functions

## Classes
### FJSSPJobGenerator
Role: generates FJSSP instances with randomized job structures and machine eligibility. 
Inheritance: none.
| Method | Signature | Description |
|--------|-----------|-------------|
| SampleNormal | `float SampleNormal(float mu, float sigma, float minValue = 1f)` | Samples a value from a normal distribution N(mu, sigma) clamped to a minimum. |
| SampleProcTime | `float SampleProcTime(MachineType type, FJSSPConfig config)` | Samples the processing time for a specific machine type based on the provided configuration. |
| Generate | `FJSSPJobDefinition[] Generate(FJSSPConfig config, Dictionary<MachineType, List<int>> machinesByType)` | Generates an array of job definitions based on the provided configuration. |
| GenerateSingle | `FJSSPJobDefinition GenerateSingle(int jobId, FJSSPConfig config, Dictionary<MachineType, List<int>> machinesByType)` | Generates a single job definition for dynamic mid-episode arrival. |
| GenerateOpSequence | `MachineType[] GenerateOpSequence(int opCount)` | Constructs a randomized sequence of machine types for a job's operations. |

## Notes
The `FJSSPJobGenerator` class uses the Box-Muller transform to generate random samples from normal distributions. The `GenerateOpSequence` method ensures that every available machine type appears at least once in the sequence and prevents consecutive duplicates of the same machine type. The class also uses `UnityEngine.Random` for random number generation, which is seeded in the `SpawnFactory`.