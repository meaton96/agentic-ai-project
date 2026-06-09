# `Simulation/Jobs/BrandimartLoader.cs`
**Language:** C# | **Lines:** 376 | **Last Scanned:** Not specified

## Purpose
This file contains the `BrandimartLoader` class, which loads Brandimarte FJSP benchmark instances from JSON and produces FJSSPJobDefinition arrays and FJSSPConfig objects. It plays a crucial role in integrating the SimulationBridge pipeline with the Brandimarte benchmark instances. The class provides methods for loading and configuring the benchmark instances, including stochastic disruption calibration.

## Dependencies
* `Newtonsoft.Json.Linq`: for parsing JSON benchmark files
* `Assets.Scripts.Simulation.Machines`: for machine type definitions
* `Assets.Scripts.Simulation.Types`: for FJSSPConfig and FJSSPJobDefinition types
* `Assets.Scripts.Simulation.Stochastic`: for stochastic disruption configuration
* `Assets.Scripts.Simulation.Logging`: for logging purposes

## Classes
### BrandimartLoader
Role: Loads and configures Brandimarte FJSP benchmark instances.
Inheritance: None (static class)
| Method | Signature | Description |
|--------|-----------|-------------|
| LoadDeferred | `(string jsonPath, int seed = 42, int agvCountOverride = -1)` | Loads a Brandimarte benchmark instance and returns a deferred configuration and job builder. |
| LoadDeferredWithStochastic | `(string jsonPath, StochasticDisruption disruption, int seed = 42, int agvCountOverride = -1)` | Loads a Brandimarte benchmark instance with calibrated stochastic disruptions. |
| LoadDeferredInternal | `(string jsonPath, int seed, StochasticDisruption disruption, int agvCountOverride = -1)` | Internal loader that parses JSON and constructs the configuration and deferred job builder. |
| BuildConfig | `(string json, string name, int seed, StochasticDisruption disruption, int agvCountOverride = -1)` | Builds the FJSSPConfig from JSON content, including stochastic parameter calibration. |
| BuildStochasticConfig | `(StochasticDisruption disruption, float meanProc, float maxProc)` | Derives calibrated stochastic parameters from instance processing statistics. |
| BuildJobs | `(string json, Dictionary<MachineType, List<int>> machinesByType)` | Builds FJSSPJobDefinition arrays from JSON content and machine type layout. |

## Notes
The `BrandimartLoader` class uses a round-robin assignment of machine types to ensure a balanced distribution of machines across types. The stochastic disruption calibration is based on the instance's processing time statistics to prevent infinite operation-restart loops. The `WEIBULL_MEAN_FACTOR` constant is used to derive the WeibullLambda parameter, which represents the mean time to failure. The `REPAIR_SIGMA` constant controls the repair duration, which is proportional to the mean processing time.