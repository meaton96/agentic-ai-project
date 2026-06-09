# `Simulation/Logging/SimLogger.cs`
**Language:** C# | **Lines:** 246 | **Last Scanned:** 2026-06-04

## Purpose
This file contains the `SimLogger` class, which provides a static utility for handling console and file-based logging within the simulation. It allows for configurable logging levels and file output, making it easier to track and debug simulation events. The logger plays a crucial role in the project by providing a centralized logging mechanism.

## Dependencies
* `UnityEngine`: for Unity-specific logging functionality (e.g., `Debug.Log`)
* `System.IO`: for file input/output operations (e.g., `File.WriteAllText`)
* `System`: for general system functionality (e.g., `Environment.GetCommandLineArgs`)

## Classes
The `SimLogger` class is a static utility class with no inheritance.
| Method | Signature | Description |
|--------|-----------|-------------|
| GetArg | `string GetArg(string name)` | Helper to extract command line arguments passed from the OS/PowerShell. |
| InitializeFileLogging | `void InitializeFileLogging(string folderPath, string baseFileName = "simulation_log")` | Configures the directory and dynamically names the file for persistent logging. |
| WriteToFile | `void WriteToFile(string message)` | Writes a message to the log file if file logging is enabled. |
| Log | `void Log(LogLevel level, string message)` | Logs a message to the console and file if the log level is enabled. |
| LogWarning | `void LogWarning(string message)` | Logs a warning message to the console and file. |
| LogError | `void LogError(string message)` | Logs an error message to the console and file. |
| Low | `void Low(string message)` | Logs a message with the Low log level. |
| Medium | `void Medium(string message)` | Logs a message with the Medium log level. |
| High | `void High(string message)` | Logs a message with the High log level. |
| Error | `void Error(string message)` | Logs a message with the Error log level. |

## Notes
The `SimLogger` class uses a static `ActiveLevel` variable to control the logging level, and a private `WriteToFile` method to handle file output. The `InitializeFileLogging` method creates the log file and sets up the file logging mechanism. The class also provides convenience methods for logging at different levels (e.g., `Low`, `Medium`, `High`, `Error`).