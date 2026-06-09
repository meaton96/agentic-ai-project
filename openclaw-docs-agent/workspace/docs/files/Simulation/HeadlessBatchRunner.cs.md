# `Simulation/HeadlessBatchRunner.cs`
**Language:** C# | **Lines:** 12000+ | **Last Scanned:** 2026-06-04

## Purpose
This file contains the `HeadlessBatchRunner` class, which drives the simulation through multiple configurations and rules in a headless environment. It is designed for data-collection builds and reads a batch configuration JSON from the command line. The class is responsible for executing a batch of simulation runs over generated job data configurations.

## Dependencies
* `Unity.MLAgents`: for machine learning agent functionality
* `Assets.Scripts.Simulation.Logging`: for logging simulation results
* `Assets.Scripts.Simulation.Machines`: for machine-related functionality
* `Assets.Scripts.Simulation.Types`: for simulation-related data types
* `Assets.Scripts.Simulation.Jobs`: for job-related functionality

## Classes
### HeadlessBatchRunner
Role: Drives the simulation through multiple configurations and rules in a headless environment.
Inheritance: `MonoBehaviour`
| Method | Signature | Description |
|--------|-----------|-------------|
| Start | `void Start()` | Initializes the batch runner and starts the simulation |
| RunBatch | `void RunBatch(FJSSPConfig[] configs, int repeats = 1)` | Starts a batch run from script or a UI button |
| RunBatchFromFile | `void RunBatchFromFile(string path, int repeats = 1)` | Starts a batch from a JSON file path |
| RunBatchCoroutine | `IEnumerator RunBatchCoroutine(FJSSPConfig[] configs, int repeats)` | Coroutine that executes a batch of simulation runs over generated job data configurations |

## Notes
The `HeadlessBatchRunner` class uses a coroutine to execute the simulation runs, allowing for asynchronous execution and efficient use of system resources. The class also uses a variety of command-line arguments to customize the simulation, such as the batch configuration file, output directory, and disruption level. The `AllRules` array defines the available dispatching rules, which can be filtered using the `-rules` command-line argument.