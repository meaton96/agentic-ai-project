# `Simulation/Channels/EpisodeTelemetryChannel.cs`
**Language:** C# | **Lines:** 276 | **Last Scanned:** 2026-06-04

## Purpose
This file implements the `EpisodeTelemetryChannel` class, which collects per-episode events and flushes them to Python as a JSON string at episode end. It plays a crucial role in logging and diagnostics for the simulation. The class enables the collection of various events, such as machine failures and repairs, and provides a way to send this data to Python for further analysis.

## Dependencies
* `Unity.MLAgents.SideChannels`: for side channel communication with Python
* `Newtonsoft.Json`: for JSON serialization of telemetry data
* `Assets.Scripts.Simulation.Logging`: for logging purposes

## Classes
### EpisodeTelemetryChannel
Role: Collects and sends episode telemetry data to Python. 
Inherits from `SideChannel`.
| Method | Signature | Description |
|--------|-----------|-------------|
| `RecordMachineFailure` | `(int, float, float)` | Records a machine failure event |
| `RecordMachineRepairComplete` | `(int, float)` | Records a machine repair completion event |
| `RecordAGVFailure` | `(int, float, float)` | Records an AGV failure event |
| `RecordJobArrival` | `(int, float)` | Records a job arrival event |
| `RecordEpisodeResult` | `(double, int, int, int, int, double, string, string)` | Records the episode result |
| `Flush` | `()` | Sends collected events and episode result to Python |

### TelemetryEvent
Role: Represents a single telemetry event.
| Method | - | Description |
|--------|---|-------------|
| - | - | No methods, only properties for event data |

### EpisodeResult
Role: Represents the result of an episode.
| Method | - | Description |
|--------|---|-------------|
| - | - | No methods, only properties for result data |

### TelemetryPayload
Role: Represents the payload sent to Python, containing events and episode result.
| Method | - | Description |
|--------|---|-------------|
| - | - | No methods, only properties for payload data |

## Notes
The `EpisodeTelemetryChannel` class uses a singleton pattern to ensure only one instance is created. The `Flush` method is used to send the collected data to Python, and it clears the event list after sending to prepare for the next episode. The class relies on JSON serialization to send data to Python, which is deserialized on the Python side for further analysis.