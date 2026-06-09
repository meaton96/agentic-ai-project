# `Simulation/Stochastic/StochasticDistributionValidator.cs`
**Language:** C# | **Lines:** 256 | **Last Scanned:** Not Available

## Purpose
This file contains a C# class `StochasticDistributionValidator` that performs statistical validation of stochastic distributions used in the simulation. It generates a large number of samples from each distribution and checks if the empirical statistics (mean, standard deviation) match the theoretical values within a certain tolerance. The class is designed to be used as a headless unit test, running in batch mode without graphics.

## Dependencies
* `UnityEngine`: for MonoBehaviour and coroutine support
* `Assets.Scripts.Simulation.Types`: for FJSSPConfig and StochasticConfig classes
* `Assets.Scripts.Simulation.Logging`: for logging functionality

## Classes
### StochasticDistributionValidator
Role: Validates stochastic distributions used in the simulation. 
Inherits from: `MonoBehaviour`.
| Method | Signature | Description |
|--------|-----------|-------------|
| Start | `void Start()` | Entry point, checks for CLI flag and launches test suite |
| RunAllTests | `IEnumerator RunAllTests()` | Executes the full test suite sequentially |
| CheckStat | `void CheckStat(string label, double actual, double expected)` | Evaluates a single statistical metric and reports PASS or FAIL |
| MakeConfig | `FJSSPConfig MakeConfig(...)` | Constructs a fully-populated FJSSPConfig for test scenarios |
| HasCLIFlag | `bool HasCLIFlag(string flag)` | Checks if a specific CLI flag is present in the process command-line arguments |

## Notes
The `StochasticDistributionValidator` class uses a coroutine to run the test suite, allowing for asynchronous execution of the tests. The `MakeConfig` method provides a convenient way to create a test configuration with default values, which can be overridden as needed. The `HasCLIFlag` method is used to check for the presence of a specific CLI flag, which determines whether the test suite should be run.