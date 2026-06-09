# `Simulation/Logging/ResultsLogger.cs`
**Language:** C# | **Lines:** 571 | **Last Scanned:** 2026-06-04

## Purpose
This file contains the `ResultsLogger` class, which is responsible for serializing `EpisodeRecord` objects to CSV files. It plays a crucial role in logging simulation results, including episode data, machine utilization, AGV performance, and segment congestion. The logger allows for customizable output file names and directories.

## Dependencies
* `System.IO`: for file input/output operations
* `UnityEngine`: for accessing application data paths
* `Assets.Scripts.Simulation.Types`: for `EpisodeRecord` and related types

## Classes
### ResultsLogger
Role: Static class for logging simulation results to CSV files. 
Inheritance: None.
| Method | Signature | Description |
|--------|-----------|-------------|
| SetFilenameSuffix | `void SetFilenameSuffix(string suffix)` | Sets the suffix for output file names. |
| SetSubdirectory | `void SetSubdirectory(string subdir)` | Sets the output directory for log files. |
| LogAll | `void LogAll(EpisodeRecord r)` | Logs all simulation results (episode, machine, AGV, segment) to their respective CSV files. |
| LogEpisode | `void LogEpisode(EpisodeRecord r)` | Logs episode data to the results CSV file. |
| LogMachineUtilization | `void LogMachineUtilization(EpisodeRecord r)` | Logs machine utilization data to the machine utilization CSV file. |
| LogAGVPerformance | `void LogAGVPerformance(EpisodeRecord r)` | Logs AGV performance data to the AGV performance CSV file. |
| LogSegmentCongestion | `void LogSegmentCongestion(EpisodeRecord r)` | Logs segment congestion data to the segment congestion CSV file. |
| BuildPath | `string BuildPath(string filename)` | Builds the full path for a given file name. |
| StripExt | `string StripExt(string name, string ext)` | Removes the specified extension from a file name. |

## Notes
The `ResultsLogger` class uses a static approach to simplify logging and reduce overhead. It relies on the `EpisodeRecord` type to encapsulate simulation data, making it easy to extend or modify logging functionality as needed. The use of CSV files allows for easy data analysis and import into external tools.