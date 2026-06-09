# `Simulation/LoggingInit.cs`
**Language:** C# | **Lines:** 78 | **Last Scanned:** 2026-06-04

## Purpose
This file is responsible for initializing the logging utilities in the simulation. It ensures that both `ResultsLogger` and `SimLogger` are configured with the correct directory paths and verbosity levels. The logging initializer handles cross-platform directory resolution to write logs to persistent storage locations outside of read-only application bundles.

## Dependencies
* `UnityEngine`: provides access to Unity's application data path and environment information.
* `Assets.Scripts.Simulation.Logging`: contains logging utilities, including `ResultsLogger` and `SimLogger`.

## Classes
The `LoggingInitializer` class orchestrates the initialization of static logging utilities. It inherits from `MonoBehaviour`.
| Method | Signature | Description |
|--------|-----------|-------------|
| `Awake` | `void Awake()` | Performs the initial setup of logging paths and configurations. |
| `GetCLIArg` | `string GetCLIArg(string key)` | Retrieves a specific value from the application's command-line arguments. |

## Notes
The `LoggingInitializer` class uses preprocessor directives to resolve the "Results" directory based on the execution environment (Editor, Windows/Linux Standalone, or macOS Standalone). It also checks for the `-loglevel` command-line argument to allow for runtime verbosity overrides. The `GetCLIArg` method iterates through the command-line arguments to find the specified key and returns the subsequent array element as the value.